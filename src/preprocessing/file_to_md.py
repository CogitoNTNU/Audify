from markitdown import MarkItDown
import os
from urllib.parse import urlparse

#A function that converts other file types to Markdown files
def file_to_md(input_file: str, output_filename: str):

    md = MarkItDown()
    parsed = urlparse(input_file)
#To get a relative file path (so we can place it correctly)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if parsed.scheme in ("http", "https"):
        md_content = md.convert(input_file)

    else: 

        #Builds the file_path from script_dir + input_file
        file_path = os.path.abspath(os.path.join(script_dir, input_file))

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fant ikke filen: {input_file}")
        
        md_content = md.convert(file_path)
    
    base_dir = os.path.join(script_dir, "..", "..", "data", "markdown")
    os.makedirs(base_dir, exist_ok=True)
    output_path = os.path.join(base_dir, output_filename)

    #Writes/creates a new file that is saved in the 'markdown' folder in Documents
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content.text_content)

