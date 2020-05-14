from __future__ import print_function
import numpy as np

"""
Cube Coordinates:

    Pointy orientation


         |
m.       |       .l
     .   |   .
        .|.
     .   |   .
 .       |       .
         |
         v
         n


        l = 0      m = 0
         \        /        /\ 
          \/\  /\/        / l\ 
          / 0\/+1\        | m| 
          |+1|| 0|        \ n/
         /\-1/\-1/\        \/
        /-1\/ 0\/+1\
        |+1 | 0 |-1 |--- n = 0
        \ 0/\ 0/\ 0/
         \/  \/ 0\/
          | 0||-1|
          \+1/\+1/
           \/  \/

    Flat orientation

  +m      -n
   \     /
    \   /
     \ /
 -----------> +l
     / \ 
    /   \ 
   /     \ 


       __            __
      / 0\          / l\ 
   __/ +1 \__      /  m \ 
  /-1\ -1 /+1\     \  n /
 / +1 \__/  0 \     \__/
 \  0 / 0\ -1 /
  \__/  0 \__/
  /-1\  0 /+1\ 
 /  0 \__/ -1 \ 
 \ +1 / 0\  0 /
  \__/ -1 \__/
     \ +1 /
      \__/

Note: a wonderful primer on hexagon grids,
https://www.redblobgames.com/grids/hexagons/

"""

def neighbors(hex_orientation="flat", coordinate_system="cube"):
    """ anti-clockwise list of neighbors starting with the highest """
    cube_neighbors = np.array([[0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0], [1, 0, -1]])
    if hex_orientation == "flat" and coordinate_system == "cube":
        return cube_neighbors
    elif hex_orientation == "flat" and coordinate_system == "axial":
        return cube_to_axial(cube_neighbors)
    elif hex_orientation == "flat" and coordinate_system == "doubled":
        return cube_flat_to_doubled_flat(cube_neighbors)
    elif hex_orientation == "flat" and coordinate_system == "xy":
        return cube_flat_to_xy(cube_neighbors)
    elif hex_orientation == "pointy" and coordinate_system == "xy":
        return cube_pointy_to_xy(cube_neighbors)
    else:
        raise NotImplementedError

## Coordinate transforms
def xy_to_hex(xy, hex_orientation="flat", coordinate_system="cube"):
    if hex_orientation == "flat" and coordinate_system == "cube":
        return xy_to_cube_flat(xy)
    else:
        raise NotImplementedError

def xy_to_cube_pointy(xy):
    """ Transform from cube (lmn) to 2D (xy) coordinates for Flat hexagon grid.

    Parameters
    ----------
      xy : ndarray
          points in xy, of shape (n_points, 2)

    Returns
    -------
      lmn : ndarray,
          points in lmn, of shape (n_points, 3)

    Example
    -------
    >>> xy_to_cube_pointy(np.array([[ 1. ,  0.5], [-1. ,  0.5]]))
    array([[ 1, -1,  0],
           [-1,  1,  0]])
    """
    # check constraints
    sqrt3 = np.sqrt(3.)
    # pointy hex coordinate transform
    XY_to_LM_mat = np.array([[ 1, 2/3],
                             [-1, 2/3]])
    lm = np.matmul(XY_to_LM_mat, xy.T).T.astype(int)
    lmn = axial_to_cube(lm)
    return lmn

def xy_to_cube_flat(xy):
    """ Transform from cube (lmn) to 2D (xy) coordinates for Flat hexagon grid.

    Parameters
    ----------
      xy : ndarray
          points in xy, of shape (n_points, 2)

    Returns
    -------
      lmn : ndarray,
          points in lmn, of shape (n_points, 3)

    Example
    -------
    >>> xy_to_cube_flat(np.array([[ 1. ,  0.5], [-1. ,  0.5]]))
    array([[ 1,  0, -1],
           [-1,  1,  0]])
    """
    # check constraints
    sqrt3 = np.sqrt(3.)
    # flat hex coordinate transform
    XY_to_LM_mat = np.array([[ 4/3,  0],
                             [-2/3,  1]])
    lm = np.matmul(XY_to_LM_mat, xy.T).T.astype(int)
    lmn = axial_to_cube(lm)
    return lmn

def hex_to_xy(lmn, hex_orientation="flat", coordinate_system="cube"):
    if hex_orientation == "flat" and coordinate_system == "cube":
        return cube_flat_to_xy(xy)
    else:
        raise NotImplementedError

def cube_flat_to_xy(lmn):
    """ Transform from 2D (xy) to cube (lmn) coordinates for Flat hexagon grid.

    Parameters
    ----------
      lmn : ndarray,
          points in lmn, of shape (n_points, 3)
          in order to be considered valid cube coordinates, lmn must satisfy l+m+n=0

    Returns
    ------
      xy : ndarray
          points in xy, of shape (n_points, 2)

    Example
    -------
    >>> cube_flat_to_xy(np.array([[1, 0, -1], [-1, 1, 0]]))
    array([[ 0.75,  0.5 ],
           [-0.75,  0.5 ]])
    """
    check_cube_constraint(lmn)
    sin30 = 0.5
    # flat hex coordinate transform (l and m are enough to specify x and y)
    LMN_to_XY_mat = np.array([[ 0.75, 0, 0],
                              [sin30, 1, 0]])
    xy = np.matmul(LMN_to_XY_mat, lmn.T).T
    return xy

def cube_pointy_to_xy(lmn):
    """ Transform from 2D (xy) to cube (lmn) coordinates for Flat hexagon grid.

    Parameters
    ----------
      lmn : ndarray,
          points in lmn, of shape (n_points, 3)
          in order to be considered valid cube coordinates, lmn must satisfy l+m+n=0

    Returns
    ------
      xy : ndarray
          points in xy, of shape (n_points, 2)

    Example
    -------
    >>> cube_pointy_to_xy(np.array([[1, 0, -1], [-1, 1, 0]]))
    array([[ 0.5 ,  0.75],
           [-1.  ,  0.  ]])
    """
    check_cube_constraint(lmn)
    sin30 = 0.5
    # pointy hex coordinate transform (l and n are enough to specify x and y)
    LMN_to_XY_mat = np.array([[ 1, 0, sin30],
                              [ 0, 0, -0.75]])
    xy = np.matmul(LMN_to_XY_mat, lmn.T).T
    return xy

def axial_to_cube(lm):
    """ Transforms from lm axial coordinates to lmn axial coordinates,
    adding the explicit n coordinate by solving the constraint l+m+n=0 """
    lmn = np.zeros((lm.shape[0], 3))
    lmn[:,:2] = lm
    lmn[:,2] = -lm[:,0] -lm[:,1]
    return lmn.astype(int)

def cube_to_axial(lmn):
    """ Transforms from lmn cube coordinates to lm axial coordinates,
    whereby the n is simply removed """
    return lmn[:,:2]

def check_cube_constraint(lmn):
    """ check constraints on cube grid l + m + n = 0 """
    if not np.allclose(np.sum(lmn, axis=-1), 0):
        raise ValueError("cube coordinates must satisfy l + m + n = 0")

def check_doubled_constraint(ij):
    """ check constraint on doubled grid (i + j) % 2 == 0 """
    if not np.allclose(np.mod(ij[:,0] + ij[:,1], 2), 0):
        raise ValueError("doubled coordinates must satisfy (i + j) % 2 == 0")

def cube_flat_to_doubled_flat(lmn):
    check_cube_constraint(lmn)
    ij = np.zeros((lmn.shape[0], 2))
    ij[:, 0] = lmn[:, 0]
    ij[:, 1] = 2* lmn[:, 1] + lmn[:,0]
    return ij

def doubled_flat_to_cube_flat(ij):
    check_doubled_constraint(ij)
    lmn = np.zeros((ij.shape[0], 3))
    lmn[:, 0] = ij[:, 0]
    lmn[:, 1] = 0.5 * ( ij[:, 1] - ij[:, 0] )
    lmn[:, 2] = -lmn[:,0] -lmn[:,1]
    return lmn

def print_ascii_hex(h_tiles=3, v_tiles=4,
        coordinate_system="cube", hex_orientation="flat",
        string_creator_func=None, legend_lines=None):
    """

    Parameters
    ----------
    h_tiles: int, >= 3
        number of tiles to apply horizontally
    v_tiles: int, >= 4
        number of tiles to apply vertically

    Notes
    -----
    Pattern is built out of the following horizontal/vertical repeated tiles:

    i : inner area of current hexagon
    b : border between current and neighbor hexagon
    o : neighbor (of current hexagon) inner area
    d : border between 2 neighbors of current hexagon

               11             11
     012345678901  0012345678901
    0/      \____  0doooooodbbbb
    1\      /      1doooooobiiii
    2 \____/       2odddddbiiiii
    3 /    \       3odoooobiiiii

    """
    # function which determines what gets printed in the center of the hexagons
    # takes two scalars (i, j), doubled_flat coordinates of the hexagon,
    # outputs a len 3 list of len 4 strings, (one len 4 string per writable line in the hexagon)
    if string_creator_func is None:
        def coordinates_to_str(i, j):
            # cube coordinates of bottom-right hexagon in current tile
            l, m, n = doubled_flat_to_cube_flat(np.array([[i, j]]))[0]
            if coordinate_system == "cube":
                coordinates_as_str = ["{:+4d}".format(int(cc))[:4] for cc in [l, m, n]]
            elif coordinate_system == "doubled":
                coordinates_as_str = ["{:+4d}".format(int(cc))[:4] for cc in [i, j]]
                coordinates_as_str.append('    ')
            elif coordinate_system == "xy":
                x, y = cube_flat_to_xy(np.array([[l,m,n]]))[0]
                coordinates_as_str = ["{:+.1f}".format(cc)[:4] for cc in [x, y]]
                coordinates_as_str.append('    ')
            else:
                raise NotImplementedError
            return coordinates_as_str
        string_creator_func = coordinates_to_str
    # overridable legend strings
    if legend_lines is None:
        if coordinate_system == "cube":
            coordinate_names = ['l', 'm', 'n']
        elif coordinate_system == "doubled":
            coordinate_names = ['i', 'j', ' ']
        elif coordinate_system == "xy":
            coordinate_names = ['x', 'y', ' ']
        else:
            raise NotImplementedError
        legend_lines = coordinate_names
    # print the tiles
    if hex_orientation=="flat":
        TILE_WIDTH = 12
        TILE_HEIGHT = 4
        # determine amount of lines and columns to print
        n_lines = v_tiles * TILE_HEIGHT
        n_cols = h_tiles * TILE_WIDTH
        for j_tile in range(v_tiles): # cycle tiles vertically
            for tile_line in range(TILE_HEIGHT): # print tile line-by-line
                for i_tile in range(h_tiles): # cycle tiles horizontally
                    for tile_col in range(TILE_WIDTH): # print line char-by-char
                        # don't print edge chars
                        if i_tile == (h_tiles - 1):
                            if tile_col in [8, 9, 10, 11]:
                                print(' ',end='')
                                continue
                        if j_tile == 0:
                            if i_tile == 0:
                                if (tile_line, tile_col) in [(1,0), (2,1)]:
                                    print(' ',end='')
                                    continue
                            if i_tile == (h_tiles - 1):
                                if (tile_line, tile_col) in [(1,7), (2,6)]:
                                    print(' ',end='')
                                    continue
                            if tile_line == 0 and  tile_col < 8:
                                print(' ',end='')
                                continue
                            if tile_col in [1,2,3,4,5,6] and tile_line in [0,1]:
                                print(' ',end='')
                                continue
                        if j_tile == (v_tiles - 2):
                            if tile_line == 3 and tile_col in [2,3,4,5]:
                                print(' ',end='')
                                continue
                            if i_tile == 0:
                                if tile_line == 3 and tile_col in [0, 1]:
                                    print(' ',end='')
                                    continue
                            if i_tile == (h_tiles - 1):
                                if (tile_line, tile_col) in [(3,6), (3,7)]:
                                    print(' ',end='')
                                    continue
                        if j_tile == (v_tiles - 1):
                            if i_tile == 0:
                                if (tile_line, tile_col) in [(0,0)]:
                                    print(' ',end='')
                                    continue
                            if i_tile == (h_tiles - 1):
                                if (tile_line, tile_col) in [(0,7)]:
                                    print(' ',end='')
                                    continue
                            if tile_col in [1,2,3,4,5,6]:
                                print(' ',end='')
                                continue
                            if tile_line in [1, 2, 3]:
                                print(' ',end='')
                                continue

                        # doubled flat hex coordinates of bottom-right hexagon in current tile
                        i = 2 * i_tile
                        j = 2 * (v_tiles - 1 - j_tile) - 4

                        # find which are current char is in (32 characters in tile)
                        # '/' chars
                        if (tile_line, tile_col) in [(0, 0), (1, 7), (2, 6), (3, 1)]:
                            print('/', end='')
                        # '\' chars
                        elif (tile_line, tile_col) in [(0, 7), (1, 0), (2, 1), (3, 6)]:
                            print('\\', end='')
                        # '_' chars
                        elif (tile_line, tile_col) in [(0,8), (0,9), (0,10), (0,11),
                                                       (2,2), (2,3), (2,4), (2,5)]:
                            print('_', end='')
                        # writable inner hex area
                        elif tile_line in [1, 2, 3] and tile_col == 8:
                            coordinate_index = tile_line - 1
                            print(string_creator_func(i,j)[coordinate_index], end='')
                        elif tile_line in [1, 2, 3] and tile_col in [9,10,11]:
                            print('', end='')
                        # writable top-left neighbor inner hex area
                        elif tile_line in [0, 1] and tile_col == 2:
                            i_tl = i - 1
                            j_tl = j + 1
                            coordinate_index = tile_line + 1
                            print(string_creator_func(i_tl,j_tl)[coordinate_index], end='')
                        elif tile_line in [0, 1] and tile_col in [3,4,5]:
                            print('', end='')
                        # writable bottom-left neighbor inner hex area
                        elif tile_line == 3 and tile_col == 2:
                            i_bl = i - 1
                            j_bl = j - 1
                            coordinate_index = 0
                            print(string_creator_func(i_bl,j_bl)[coordinate_index], end='')
                        elif tile_line == 3 and tile_col in [3,4,5]:
                            print('', end='')
                        # inner hex area
                        elif (tile_line in [2, 3] and tile_col == 7):
                            print(' ', end='')
                        # inner hex area of neighbors
                        elif (tile_line in [2, 3] and tile_col == 0):
                            print(' ', end='')
                        elif (tile_line in [0, 1] and tile_col in [1, 6]):
                            print(' ', end='')
                        elif (tile_line in [0, 1] and tile_col == 3):
                            print(' ', end='')
                        # errors (undefined locations)
                        else:
                            print('x', end='')

                    if i_tile == (h_tiles - 1):
                        # legend
                        if j_tile == 0:
                            print([
                            "  ____   ",
                            " /{:>4}\  ".format(legend_lines[0][:4]),
                            "/ {:>4} \ ".format(legend_lines[1][:4]),
                            "\ {:>4} / ".format(legend_lines[2][:4]),
                            ][tile_line])
                        elif j_tile == 1 and tile_line == 0:
                            print(" \____/  ")
                        # linebrea
                        else:
                            print()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
