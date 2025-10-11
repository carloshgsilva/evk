
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cctype>

#pragma comment(lib, "dbghelp.lib")
// Custom exception handler
LONG WINAPI CustomUnhandledExceptionFilter(EXCEPTION_POINTERS* pExceptionInfo) {
    std::cerr << "Fatal error: Unhandled exception occurred." << std::endl;

    // Generate the stack trace
    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();
    CONTEXT context = *pExceptionInfo->ContextRecord;

    SymInitialize(hProcess, NULL, TRUE);

    STACKFRAME64 stack_frame;
    ZeroMemory(&stack_frame, sizeof(STACKFRAME64));
    stack_frame.AddrPC.Offset = context.Rip;
    stack_frame.AddrPC.Mode = AddrModeFlat;
    stack_frame.AddrFrame.Offset = context.Rbp;
    stack_frame.AddrFrame.Mode = AddrModeFlat;
    stack_frame.AddrStack.Offset = context.Rsp;
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

    for (DWORD i = 0; StackWalk64(IMAGE_FILE_MACHINE_AMD64, hProcess, hThread, &stack_frame, &context, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL) && (i < 50); ++i) {
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
            if (fileLower.find("program files", 0) == -1 && pSymbol->Name) {
                std::cerr << "-> " << pSymbol->Name << "() [" << lineInfo.FileName << ":" << std::dec << (lineInfo.LineNumber) << "]" << std::endl;
            }
        }
    }

    SymCleanup(hProcess);

    return EXCEPTION_CONTINUE_SEARCH; // Pass to next handler
}

void set_unhandled_exception_filter() {
    SetUnhandledExceptionFilter(CustomUnhandledExceptionFilter);
}
