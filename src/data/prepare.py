"""Download LT-OPD-14K, verify every image, and materialize the training parquet."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def file_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def sha256(path: Path) -> str:
    return file_fingerprint(path)[1]


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Invalid release path: {value}')
    return Path(*path.parts)


def check_image(path: Path, item: dict) -> bool:
    try:
        size, digest = file_fingerprint(path)
    except (FileNotFoundError, IsADirectoryError):
        return False
    return size == item['size_bytes'] and digest == item['sha256']


def extract_archive(archive: Path, output: Path, expected: dict[str, dict]) -> None:
    seen = set()
    with tarfile.open(archive, 'r') as tar:
        for member in tar:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f'Unexpected archive member: {member.name}')
            seen.add(member.name)
            item = expected[member.name]
            if member.size != item['size_bytes']:
                raise ValueError(f'Archive size mismatch: {member.name}')
            target = output / safe_relative(member.name)
            if check_image(target, item):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + '.part')
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f'Unreadable archive member: {member.name}')
            with source, temp.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            if not check_image(temp, item):
                temp.unlink(missing_ok=True)
                raise ValueError(f'Image hash mismatch: {member.name}')
            temp.replace(target)
    if seen != set(expected):
        raise ValueError(f'Archive is missing {len(set(expected) - seen)} expected images: {archive.name}')


def from_sources(item: dict, roots: list[Path], output: Path) -> bool:
    target = output / safe_relative(item['path'])
    if check_image(target, item):
        return True
    source_key = safe_relative(item['source_media_key'])
    names = [safe_relative(item['path']), Path(item['path']).name, source_key, source_key.name]
    for root in roots:
        for name in names:
            source = root / name
            if not check_image(source, item):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + '.part')
            temp.unlink(missing_ok=True)
            try:
                os.link(source, temp)
            except OSError:
                shutil.copyfile(source, temp)
            temp.replace(target)
            return True
    return False


def materialize(release: Path, output: Path, roots: list[Path]) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((release / 'manifest.json').read_text())
    if manifest['train_rows'] != 14000:
        raise ValueError('This training recipe expects exactly 14,000 samples')
    print('Verifying release files', flush=True)
    for name, detail in manifest['files'].items():
        path = release / safe_relative(name)
        size, digest = file_fingerprint(path)
        if size != detail['bytes'] or digest != detail['sha256']:
            raise ValueError(f'Release file verification failed: {name}')
    items = [json.loads(line) for line in (release / 'media.jsonl').read_text().splitlines()]
    if len(items) != 14000 or len({x['sample_uid'] for x in items}) != 14000:
        raise ValueError('Invalid media manifest')
    # An already-verified image directory can be reused instead of extracting the archives.
    missing = []
    verified = set()
    if roots:
        for index, item in enumerate(items, start=1):
            if from_sources(item, roots, output):
                verified.add(item['path'])
            else:
                missing.append(item)
            if index % 2000 == 0:
                print(f'Resolving images: {index}/14000', flush=True)
    else:
        missing = list(items)
    for name in sorted({item['archive'] for item in missing if item.get('archive')}):
        expected = {item['path']: item for item in items if item.get('archive') == name}
        print(f'Extracting {name}', flush=True)
        extract_archive(release / safe_relative(name), output, expected)
        verified.update(expected)
    missing = [item for item in items if item['path'] not in verified]
    if missing:
        missing_file = output / 'missing_media.jsonl'
        missing_file.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in missing))
        counts = dict(Counter(item['access'] for item in missing))
        print(f'Missing images: {counts}. Details: {missing_file}')
        print('The release archives did not contain every image. Re-download the release and retry:')
        print('all 14,000 images ship in images/part-*.tar and are verified against media.jsonl.')
        raise SystemExit(2)
    table = pq.read_table(release / 'train.parquet')
    rows = table.to_pylist()
    if len(rows) != 14000:
        raise ValueError('Unexpected training row count')
    if len({row['sample_uid'] for row in rows}) != 14000:
        raise ValueError('Duplicate sample_uid values in the training set')
    for index, (row, item) in enumerate(zip(rows, items, strict=True)):
        if row['sample_uid'] != item['sample_uid'] or item['index'] != index:
            raise ValueError(f'Media identity mismatch at row {index}')
        profiles = (row['extra_info'] or {}).get('visual_capacity_profiles')
        if not profiles:
            raise ValueError(f'Row {index} carries no visual capacity profile for rollout packing')
        for name, profile in profiles.items():
            for field in ('max_pixels', 'dense_visual_tokens', 'dense_teacher_prompt_tokens',
                          'merged_student_prompt_tokens', 'raw_patch_tokens_per_view'):
                value = profile.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f'Capacity profile {name} at row {index} has an invalid {field}: {value!r}')
        if row['images'] != row['teacher_images'] or row['images'] != [{'path': item['path']}]:
            raise ValueError(f'Student/teacher image mismatch at row {index}')
        for field in ('images', 'teacher_images'):
            row[field] = [{'path': str((output / safe_relative(item['path'])).resolve())}]
    dest = output / 'train.parquet'
    temporary = output / 'train.parquet.part'
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
    if pq.read_table(temporary).to_pylist() != rows:
        raise ValueError('Parquet round-trip check failed')
    temporary.replace(dest)
    prepared = dict(manifest)
    prepared['train_sha256'] = sha256(dest)
    prepared['dataset_repository'] = 'yyy051007/LT-OPD-14K'
    (output / 'manifest.json').write_text(json.dumps(prepared, indent=2) + '\n')
    print(f'Ready: {dest} (14,000 samples, all image hashes verified)', flush=True)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='yyy051007/LT-OPD-14K')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', type=Path, default=Path('data/LT-OPD-14K'))
    parser.add_argument('--release-dir', type=Path, help='Use an already downloaded release')
    parser.add_argument('--source-media', type=Path, action='append', default=[],
                        help='Image directory to reuse instead of extracting; repeat for multiple roots')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.release_dir:
        release = args.release_dir.resolve()
        if release == output:
            parser.error('--release-dir and --output must be different')
    else:
        from huggingface_hub import snapshot_download
        release = Path(snapshot_download(repo_id=args.dataset, repo_type='dataset', revision=args.revision,
                       local_dir=output / '_release', allow_patterns=['train.parquet', 'manifest.json', 'media.jsonl', 'images/*.tar']))
    materialize(release, output, [p.resolve() for p in args.source_media])


if __name__ == '__main__':
    main()
