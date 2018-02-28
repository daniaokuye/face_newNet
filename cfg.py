# resize img
HW_h, HW_w = 800, 800
Norm_mean = [0.485, 0.456, 0.406]
Norm_std = [0.229, 0.224, 0.225]
# total stride between ground truth & predicted box
stride_all = 8
# sleep_time = 0.5
total_thread = 1000  # total number of threading
pixels_size = 32.0  # pixels to differ big or tiny


class Global_var():
    # big_now meaning that trainning big face right now
    use_skip = False


def set_big(status):
    Global_var.use_skip = status


def get_status():
    return Global_var.use_skip

# //////////////////////////////////////////

# if used_layer[i][0] == 0 or used_layer[i][1] == 0:
#     used_layer.pop(i)

# multi threading
# is_multi_thred = False if len(used_layer.keys()) < 2 + 8 else True
# res = []
# for i in used_layer.keys():
#     if not isinstance(i, int):
#         continue
#     if is_multi_thred:
#         t = threading.Thread(target=trans_used_layer,
#                              args=(H_W_factor, used_layer[i]))
#         res.append(t)
#     else:
#         trans_used_layer(H_W_factor, used_layer[i])
# for t in res:
#     t.setDaemon(True)
#     t.start()
# if is_multi_thred:
#     res[-1].join()
