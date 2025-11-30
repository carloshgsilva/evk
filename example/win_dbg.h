
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif
#include <fstream>
#include <string>
#include <cctype>

#pragma comment(lib, "dbghelp.lib")
// Custom exception handler
static void PrintStackTraceFromContext(const CONTEXT &context) {
    // Generate the stack trace using the supplied context
    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();
    CONTEXT ctx = context;

    SymInitialize(hProcess, NULL, TRUE);

    STACKFRAME64 stack_frame;
    ZeroMemory(&stack_frame, sizeof(STACKFRAME64));
    stack_frame.AddrPC.Offset = ctx.Rip;
    stack_frame.AddrPC.Mode = AddrModeFlat;
    stack_frame.AddrFrame.Offset = ctx.Rbp;
    stack_frame.AddrFrame.Mode = AddrModeFlat;
    stack_frame.AddrStack.Offset = ctx.Rsp;
    stack_frame.AddrStack.Mode = AddrModeFlat;

    // Build a lowercase current-directory prefix so we only log source lines
    // for files inside the current project directory (avoid system libs)
    char cwdBuf[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, cwdBuf);
    std::string cwdStr = cwdBuf;
    if (!cwdStr.empty() && cwdStr.back() != '\\') cwdStr.push_back('\\');
    auto toLower = [](std::string s) {
        for (auto &c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    std::string cwdLower = toLower(cwdStr);

    for (DWORD i = 0; StackWalk64(IMAGE_FILE_MACHINE_AMD64, hProcess, hThread, &stack_frame, &ctx, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL) && (i < 50); ++i) {
        DWORD64 displacement64;
        char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
        PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)buffer;
        pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        pSymbol->MaxNameLen = MAX_SYM_NAME;

        if (!SymFromAddr(hProcess, stack_frame.AddrPC.Offset, &displacement64, pSymbol)) {
            continue;
        }

        // Try to get source file and line number for this address
        IMAGEHLP_LINE64 lineInfo;
        DWORD displacementLine = 0;
        ZeroMemory(&lineInfo, sizeof(IMAGEHLP_LINE64));
        lineInfo.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        if (SymGetLineFromAddr64(hProcess, stack_frame.AddrPC.Offset, &displacementLine, &lineInfo)) {
            // Only log file/line if the source file is inside the current project directory
            std::string fileName = lineInfo.FileName ? lineInfo.FileName : std::string();
            std::string fileLower = toLower(fileName);
            if ((fileLower.find("program files", 0) == -1 || fileLower.find(cwdLower) != std::string::npos) && pSymbol->Name) {
                // Skip internal helper frames
                if (strcmp(pSymbol->Name, "crtReportHook") == 0 || strcmp(pSymbol->Name, "CustomUnhandledExceptionFilter") == 0) {
                    continue;
                }
                printf("-> %s() [%s:%d]\n", pSymbol->Name, lineInfo.FileName, lineInfo.LineNumber);
            }
        }
        // Stop at main (don't show internal init functions)
        if (strcmp(pSymbol->Name, "main") == 0) {
            break;
        }
    }

    SymCleanup(hProcess);
}

#ifdef _MSC_VER
#ifdef _DEBUG
// CRT report hook implementation. The hook will capture the context and print a stack trace
static int __cdecl crtReportHook(int reportType, char* message, int* returnValue) {
    if (reportType == _CRT_ASSERT || reportType == _CRT_ERROR || reportType == _CRT_WARN) {
        if (message) {
            printf("\n%s\n", message);
        }
        CONTEXT ctx;
        RtlCaptureContext(&ctx);
        PrintStackTraceFromContext(ctx);
        // If a debugger is attached, break to allow inspection; otherwise exit
        fflush(stderr);
        if (IsDebuggerPresent()) {
            DebugBreak();
        } else {
            exit(1);
        }
        return 1; // handled
    }
    return 0; // not handled
}
#endif
#endif

LONG WINAPI CustomUnhandledExceptionFilter(EXCEPTION_POINTERS* pExceptionInfo) {
    printf("Fatal error: Unhandled exception occurred.\n");

    // Generate the stack trace
    CONTEXT context = *pExceptionInfo->ContextRecord;
    PrintStackTraceFromContext(context);

    exit(1);
    return EXCEPTION_EXECUTE_HANDLER;
}

// (Hook defined above)

void set_unhandled_exception_filter() {
    // SetUnhandledExceptionFilter(CustomUnhandledExceptionFilter);
#ifdef _MSC_VER
    // Hook CRT reports (assert/warn/error) in debug builds
#ifdef _DEBUG
    // Install the CRT report hook (stores previous hook if needed)
    _CrtSetReportHook(crtReportHook);
#endif
#endif
}
