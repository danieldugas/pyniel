from hexgrid import *

print("Flat orientation grid examples")
print("")
print("  Cube coordinates:")
print("")
print_ascii_hex(coordinate_system="cube")
print("  Doubled coordinates:")
print("")
print_ascii_hex(v_tiles=10, h_tiles=6, coordinate_system="doubled")
print("  xy coordinates:")
print("")
print_ascii_hex(coordinate_system="xy")

print("  Arbitrary text:")
print("")
def string_creator_func(i, j):
    return ["    ",
            "o o ",
            " w  "]
print_ascii_hex(string_creator_func=string_creator_func, legend_lines=string_creator_func(0,0))
