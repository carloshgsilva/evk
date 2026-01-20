## Build / Test / Run

To build, test or run the project call `.\run.bat` from a Windows terminal (cmd or PowerShell).

The runner forwards any command-line arguments to the built executable and supports the following flags:

- `--test`  : run the test suite
- `--bench` : run the benchmarks
- `--llm`   : run the LLM demo (`main_llm.cpp`)

Flags can be combined (e.g. `--bench --llm`). If no flags are provided the script defaults to running the benchmarks and tests (preserves previous behavior).

Examples:

```
.\run.bat --bench        # run only benchmarks
.\run.bat --test --bench # run tests and benchmarks
.\run.bat --llm          # run the LLM demo
```

Note: `run.bat` will configure, build the project, compile shaders and then run the application with the supplied flags.