from markitdown import MarkItDown
import os
from urllib.parse import urlparse

#En fnunksjon som konverter andre fil_typer til md-filer
def file_to_md(input_file: str, output_filename: str):

    md = MarkItDown()
    parsed = urlparse(input_file)
    #for å hente en relativ filsti (slik at vi kan plassere den riktig)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if parsed.scheme in ("http", "https"):
        md_content = md.convert(input_file)

    else: 

        #bygger file_path ut fra script_dir + input_file
        file_path = os.path.abspath(os.path.join(script_dir, input_file))

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fant ikke filen: {input_file}")
        
        md_content = md.convert(file_path)
    
    base_dir = os.path.join(script_dir, "..", "..", "data", "processed")
    os.makedirs(base_dir, exist_ok=True)
    output_path = os.path.join(base_dir, output_filename)

    #skirv/lager ny fil som blir lagt til i processed mappe i documents
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content.text_content)

