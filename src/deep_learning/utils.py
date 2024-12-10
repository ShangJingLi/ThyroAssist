import time
import mindspore


class StopTimeMonitor(mindspore.Callback):
    def __init__(self, run_time):
        super(StopTimeMonitor, self).__init__()
        self.run_time = run_time

    def on_train_begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()  # 获取开始训练的时间

    def on_train_step_end(self, run_context):
        """每个step结束后执行的操作"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num  # 获取epoch值
        step_num = cb_params.cur_step_num  # 获取step值
        loss = cb_params.net_outputs  # 获取损失值loss
        cur_time = time.time()  # 获取当前时间戳

        if (cur_time - cb_params.init_time) > self.run_time:
            # 当训练时间达到规定的停止时间时，停止训练
            train_time = get_time(cur_time, cb_params.init_time)
            print(f"训练中止！ 用时: {train_time}, 当前epoch: {epoch_num}, 当前: {step_num}, loss:{loss}")
            run_context.request_stop()


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
           'restore_ckpt',
           'StopTimeMonitor']
