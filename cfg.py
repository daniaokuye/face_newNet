# resize img
HW_h, HW_w = 800, 800
Norm_mean = [0.485, 0.456, 0.406]
Norm_std = [0.229, 0.224, 0.225]
# total stride between ground truth & predicted box
stride_all = 8
# sleep_time = 0.5
total_thread = 1000  # total number of threading


class Global_var():
    # big_now meaning that trainning big face right now
    use_skip = False


def set_big(status):
    Global_var.use_skip = status


def get_status():
    return Global_var.use_skip
