# convert_pgm_to_png.py
from pathlib import Path
from PIL import Image

def convert_pgm_to_png():
    """
    Convierte todos los archivos .pgm del directorio actual
    a formato .png y los guarda en la carpeta 'png_outputs'.
    """
    root_dir = Path(__file__).parent        # raíz del proyecto
    output_dir = root_dir / "png_outputs"   # carpeta de salida
    output_dir.mkdir(exist_ok=True)

    for pgm_file in root_dir.glob("*.pgm"):
        try:
            img = Image.open(pgm_file)
            png_path = output_dir / (pgm_file.stem + ".png")
            img.save(png_path)
            print(f"{pgm_file.name} → {png_path.name}")
        except Exception as e:
            print(f"Error al convertir {pgm_file.name}: {e}")

if __name__ == "__main__":
    convert_pgm_to_png()
