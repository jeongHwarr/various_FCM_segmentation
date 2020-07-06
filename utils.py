def makedirs(path):
    import os
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)