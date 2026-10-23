"""GIM's demo padding: an image padded to the matcher's aspect ratio."""


def get_padding_size(image, h, w):
    orig_width = image.shape[3]
    orig_height = image.shape[2]
    aspect_ratio = w / h

    new_width = max(orig_width, int(orig_height * aspect_ratio))
    new_height = max(orig_height, int(orig_width / aspect_ratio))

    pad_height = new_height - orig_height
    pad_width = new_width - orig_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return orig_width, orig_height, pad_left, pad_right, pad_top, pad_bottom
