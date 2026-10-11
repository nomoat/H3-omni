#!/usr/bin/env python3
"""Fetch the pinned ThunderKittens headers before allocating a GPU."""
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request

REVISION = '6c27e28c8115d1839d9eeeb530913c184a75fc87'
DIGEST = '672e674b5adc63be322a3c69a7b9fc932f1e38904883c91615c54ad98c2c1d26'


def main():
    url = f'https://codeload.github.com/HazyResearch/ThunderKittens/tar.gz/{REVISION}'
    data = urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != DIGEST:
        raise RuntimeError('ThunderKittens archive digest mismatch')
    root = Path(__file__).resolve().parents[1] / '.deps/thunderkittens'
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive:
            path = Path(*Path(item.name).parts[1:])
            if not item.isfile() or '..' in path.parts or path.is_absolute():
                continue
            if str(path).startswith('include/') or str(path) == 'LICENSE':
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(item).read())
    (root / 'REVISION').write_text(REVISION + '\n')
    print(root)


if __name__ == '__main__':
    main()
