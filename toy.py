from collections import defaultdict


def remove_tex_dup(file):
    obj_file = open(file, 'r')

    vertex_list = []
    texture_list = []

    final_vertex_list = []
    final_texture_list = []

    vt_dict = defaultdict(lambda : [])

    for line in obj_file:
        split = line.split()
        # if blank line, skip
        if not len(split):
            continue
        if split[0] == "v":
            vertex_list.append(split[1:])
        elif split[0] == "vt":
            texture_list.append(split[1:])
        elif split[0] == "f":
            count = 1
            first_set = []
            second_set = []
            first_texture_set = []
            second_texture_set = []
            while count < 4:
                remove_slash = split[count].split('/')
                vt_dict[int(remove_slash[0]) - 1].append(int(remove_slash[1]) - 1)

                count += 1
            final_vertex_list.append(first_set)
            final_vertex_list.append(second_set)
            final_texture_list.append(first_texture_set)
            final_texture_list.append(second_texture_set)

    obj_file.close()
    return vt_dict


if __name__ == '__main__':
    import os
    import numpy as np
    vt_dict = remove_tex_dup('/home/justanhduc/Downloads/SMPL_UV_Parsing/smpl_part/smpl_uv.obj')
    np.save('vt_dict', dict(vt_dict))
    pass
