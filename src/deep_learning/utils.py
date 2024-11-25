import time


def get_time(start: time.time, end:time.time):
    run_time = end - start
    if int(run_time // 3600) != 0:
        hours = f"{int(run_time // 3600)}小时"
    else:
        hours = ''
    if int(int(run_time) % 3600 // 60) != 0:
        minutes = f'{int(int(run_time) % 3600 // 60)}分钟'
    else:
        minutes = ''
    if int(run_time) % 60 != 0:
        seconds = f'{run_time % 60:.2f}秒'
    else:
        seconds = ''
    return hours + minutes + seconds


def split_ckpt(file_path, chunk_size=80 * 1024 * 1024):
    """将文件分割为多个小文件，每个文件大小为chunk_size（默认为80MB）。"""
    with open(file_path, 'rb') as file:
        chunk_index = 0
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            chunk_filename = f"{file_path}.part{chunk_index:03d}"
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_index += 1

    print(f"文件分割完成，生成了{chunk_index}个部分。")


def restore_ckpt(part_prefix, output_file):
    """将分割的小文件恢复为原始文件。"""
    chunk_index = 0
    with open(output_file, 'wb') as outfile:
        while True:
            chunk_filename = f"{part_prefix}.part{chunk_index:03d}"
            try:
                with open(chunk_filename, 'rb') as infile:
                    outfile.write(infile.read())
                chunk_index += 1
            except FileNotFoundError:
                break  # 没有更多的部分文件，退出循环

    print("文件恢复完成！")


__all__ = ['get_time',
           'split_ckpt',
           'restore_ckpt']
