"""Copy a file with F_NOCACHE on both fds so the destination inode has no
pages in the unified buffer cache — gives honest cold-SSD read benchmarks
without sudo purge."""
import fcntl, os, sys

F_NOCACHE = 48  # macOS

src_path, dst_path = sys.argv[1], sys.argv[2]
src = os.open(src_path, os.O_RDONLY)
dst = os.open(dst_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
fcntl.fcntl(src, F_NOCACHE, 1)
fcntl.fcntl(dst, F_NOCACHE, 1)
total = 0
while True:
    buf = os.read(src, 8 << 20)
    if not buf:
        break
    os.write(dst, buf)
    total += len(buf)
os.close(src)
os.close(dst)
print(f"copied {total / 1e9:.2f} GB (nocache)")
