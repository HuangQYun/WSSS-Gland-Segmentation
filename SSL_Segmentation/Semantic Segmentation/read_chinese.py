import os
import openslide
import tempfile


def getinfo_from_slide(slide_path):
    # 获取文件名，不包括扩展名
    # 创建临时文件副本
    try:
        slide = openslide.open_slide(slide_path)
    except:
        slide = open_file_with_chinese_path(slide_path)
    return slide


def open_file_with_chinese_path(slide_path):
    suff_name = os.path.splitext(os.path.basename(slide_path))[1]
    with tempfile.NamedTemporaryFile(delete=True, suffix=suff_name) as tmpfile_slide:
        tmp_link_slide = tmpfile_slide.name
        if os.path.exists(tmp_link_slide):
            os.remove(tmp_link_slide)
        os.symlink(slide_path, tmp_link_slide)
        if suff_name == ".mrxs":
            filepath_fold = slide_path.split(suff_name)[0]
            tmp_link_slide_fold = tmp_link_slide.split(suff_name)[0]
            if os.path.exists(tmp_link_slide_fold):
                os.remove(tmp_link_slide_fold)
            os.symlink(filepath_fold, tmp_link_slide_fold)
            slide = openslide.OpenSlide(tmp_link_slide)
        else:
            slide = openslide.OpenSlide(tmp_link_slide)
    return slide
