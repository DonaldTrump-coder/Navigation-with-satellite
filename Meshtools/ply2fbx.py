import pyassimp

def ply2fbx(ply_file: str):
    with pyassimp.load(ply_file) as mesh:
        pyassimp.export(mesh, ply_file.replace('.ply', '.fbx'), file_type='fbx')
        print(f"Mesh saved as {ply_file.replace('.ply', '.fbx')}")