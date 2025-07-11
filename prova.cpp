
    int main(int argc, char **argv)
    {
    printf("[Matrix Multiply Using CUDA] - Starting...");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)");
        printf("  Note: Outer matrix dimensions of A & B matrices"
               " must be equal.");

        exit(EXIT_SUCCESS);
    }
    }

    