import os

def make_dir_if_not_exists(dir_):
    try: 
        os.makedirs(dir_)
    except OSError:
        if not os.path.isdir(dir_):
            raise

def crawl_subdirs_for_files(dir_, extension=".jpg"):
    files = []
    for dirpath, dirnames, filenames in os.walk(dir_):
        for filename in [f for f in filenames if f.endswith(extension)]:
            files.append(os.path.join(dirpath, filename))
    return files
