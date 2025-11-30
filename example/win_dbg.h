#ifndef WIN_DBG_H
#define WIN_DBG_H

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#include <fstream>
#include <string>
#include <cctype>

#pragma comment(lib, "dbghelp.lib")

// Custom exception handler for Windows
LONG WINAPI CustomUnhandledExceptionFilter(EXCEPTION_POINTERS* pExceptionInfo) {
    // Print exception code and a short human-readable description
    DWORD exceptionCode = pExceptionInfo && pExceptionInfo->ExceptionRecord ? pExceptionInfo->ExceptionRecord->ExceptionCode : 0;
    auto exceptionDescription = [&]() -> const char* {
        switch (exceptionCode) {
            case EXCEPTION_ACCESS_VIOLATION: return "Access violation (invalid memory access)";
            case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "Array bounds exceeded";
            case EXCEPTION_BREAKPOINT: return "Breakpoint";
            case EXCEPTION_DATATYPE_MISALIGNMENT: return "Datatype misalignment";
            case EXCEPTION_FLT_DENORMAL_OPERAND: return "Floating-point denormal operand";
            case EXCEPTION_FLT_DIVIDE_BY_ZERO: return "Floating-point divide by zero";
            case EXCEPTION_FLT_INEXACT_RESULT: return "Floating-point inexact result";
            case EXCEPTION_FLT_INVALID_OPERATION: return "Floating-point invalid operation";
            case EXCEPTION_FLT_OVERFLOW: return "Floating-point overflow";
            case EXCEPTION_FLT_STACK_CHECK: return "Floating-point stack check";
            case EXCEPTION_FLT_UNDERFLOW: return "Floating-point underflow";
            case EXCEPTION_ILLEGAL_INSTRUCTION: return "Illegal instruction";
            case EXCEPTION_IN_PAGE_ERROR: return "In-page error (paging/file I/O)";
            case EXCEPTION_INT_DIVIDE_BY_ZERO: return "Integer divide by zero";
            case EXCEPTION_INT_OVERFLOW: return "Integer overflow";
            case EXCEPTION_INVALID_DISPOSITION: return "Invalid disposition";
            case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "Noncontinuable exception";
            case EXCEPTION_PRIV_INSTRUCTION: return "Privileged instruction";
            case EXCEPTION_STACK_OVERFLOW: return "Stack overflow";
            default: return "Unknown exception";
        }
    }();

    std::cerr << "Fatal error: Unhandled exception occurred. 0x" << std::hex << exceptionCode << std::dec << " - " << exceptionDescription << std::endl;

    // Generate the stack trace
    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();
    CONTEXT context = *pExceptionInfo->ContextRecord;

    SymInitialize(hProcess, NULL, TRUE);

    STACKFRAME64 stack_frame;
    ZeroMemory(&stack_frame, sizeof(STACKFRAME64));
#if defined(_M_X64) || defined(__x86_64__)
    stack_frame.AddrPC.Offset = context.Rip;
    stack_frame.AddrFrame.Offset = context.Rbp;
    stack_frame.AddrStack.Offset = context.Rsp;
#else
    stack_frame.AddrPC.Offset = context.Eip;
    stack_frame.AddrFrame.Offset = context.Ebp;
    stack_frame.AddrStack.Offset = context.Esp;
#endif
    stack_frame.AddrPC.Mode = AddrModeFlat;
    stack_frame.AddrFrame.Mode = AddrModeFlat;
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
            if (fileLower.find("vctools", 0) == -1 && fileLower.find("msvc", 0) == -1 && pSymbol->Name) {
                std::cerr << "-> " << pSymbol->Name << "() [" << lineInfo.FileName << ":" << std::dec << (lineInfo.LineNumber) << "]" << std::endl;
            }
        }
    }

    SymCleanup(hProcess);

    exit(1);

    return EXCEPTION_CONTINUE_SEARCH; // Pass to next handler
}

void set_unhandled_exception_filter() {
    SetUnhandledExceptionFilter(CustomUnhandledExceptionFilter);
}

#elif defined(__APPLE__) || defined(__MACH__) || defined(__linux__)
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <iostream>
#include <string>

static void print_backtrace() {
    const int max_frames = 100;
    void* frames[max_frames];
    int frame_count = backtrace(frames, max_frames);
    char** symbols = backtrace_symbols(frames, frame_count);
    if (symbols) {
        for (int i = 0; i < frame_count; ++i) {
            std::cerr << symbols[i] << std::endl;
        }
        free(symbols);
    }
}

static void signal_handler(int signo, siginfo_t* info, void* context) {
    (void)info;
    (void)context;
    std::cerr << "Fatal error: signal " << signo << " received" << std::endl;
    print_backtrace();
    // Re-raise default handler to produce core dump if desired
    signal(signo, SIG_DFL);
    raise(signo);
}

void set_unhandled_exception_filter() {
    struct sigaction sa;
    sa.sa_sigaction = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
}

#else
// Fallback: no-op
#include <iostream>
void set_unhandled_exception_filter() {
    std::cerr << "Warning: no unhandled exception filter available for this platform" << std::endl;
}
#endif

#endif // WIN_DBG_H
