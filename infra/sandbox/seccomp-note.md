# Sandbox policy

Generated experiments run as a non-root user with:

- no network access by default;
- a read-only root filesystem;
- two CPU cores and 4 GB memory;
- a 128-process limit;
- a size-limited temporary filesystem;
- only the generated experiment directory copied into the image.

The local executor refuses requests that enable network access. Large or GPU-heavy jobs are
exported as reproducible packages instead of being submitted to a paid cloud service.

