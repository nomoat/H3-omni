"""Download only the pinned upstream helpers used by the VSA oracle."""
import hashlib
from pathlib import Path
import urllib.request

REVISION = 'd5253fd71055927f12aa6c4aa679b93a00238851'
FILES = {
    'video_sparse_attn.py': '6f05e429129bb131532f42e24aa868bd7752f6600bc958ea66934b6390157fc2',
    'video_sparse_attn_h3.py': 'cb08b7c40a5e33d64643cd4e052b26c9c71d0a17531faabb53b70c107698c105',
}


def main():
    root = Path(__file__).resolve().parents[2] / '.deps/fasth3-reference'
    prefix = 'fastvideo/attention/backends'
    for name, digest in FILES.items():
        url = f'https://raw.githubusercontent.com/hao-ai-lab/FastVideo/{REVISION}/{prefix}/{name}'
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'upstream digest mismatch: {name}')
        target = root / prefix / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (root / 'REVISION').write_text(REVISION + '\n')
    print(root)


if __name__ == '__main__':
    main()
