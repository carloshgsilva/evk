#if defined(_DEBUG) || defined(EVK_DEBUG)
#define EVK_ASSERT(cond, message, ...)                                                    \
    if (!(cond)) {                                                                        \
        printf("\033[1;33m" __FUNCTION__ "() \033[1;31m" message "\033[0m", __VA_ARGS__); \
        abort();                                                                          \
    }

#define CHECK_VK(cmd) EVK_ASSERT(cmd == VK_SUCCESS, #cmd)  // printf("%s\n", #cmd);
#else
#define EVK_ASSERT(cond, message, ...)
#define CHECK_VK(cmd) cmd
#endif