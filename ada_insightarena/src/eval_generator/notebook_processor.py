import nbformat
import os
import base64
from io import BytesIO
from PIL import Image

class NotebookProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.notebook = self.load_notebook()

    def load_notebook(self):
        # Load the notebook (ipynb file)
        for file in os.listdir(self.directory):
            if file.endswith('.ipynb'):
                with open(os.path.join(self.directory, file), 'r') as f:
                    return nbformat.read(f, as_version=4)
        raise FileNotFoundError("No Jupyter notebook file found in the directory")

    def process_notebook(self):
        cell_batches = []
        current_batch = []
        markdown_cells = []
        cell_number = 1

        for cell in self.notebook.cells:
            if cell.cell_type == 'markdown':
                # Remove images from markdown text
                cell.source = self.remove_images_from_text(cell.source)
                markdown_cells.append((cell_number, cell))
            elif cell.cell_type == 'code':
                if current_batch:
                    cell_with_images = self.extract_images_from_output(cell)
                    current_batch.append((cell_number, cell_with_images))
                    if cell.outputs:
                        # Add markdown cells before the first input cell
                        if markdown_cells:
                            current_batch = markdown_cells + current_batch
                            markdown_cells = []
                        # Add markdown cells after the last output cell
                        current_batch.extend(markdown_cells)
                        markdown_cells = []
                        cell_batches.append(current_batch)
                        current_batch = []
                else:
                    cell_with_images = self.extract_images_from_output(cell)
                    current_batch.append((cell_number, cell_with_images))
                    if cell.outputs:
                        # Add markdown cells before the first input cell
                        if markdown_cells:
                            current_batch = markdown_cells + current_batch
                            markdown_cells = []
                        # Add markdown cells after the last output cell
                        current_batch.extend(markdown_cells)
                        markdown_cells = []
                        cell_batches.append(current_batch)
                        current_batch = []
            cell_number += 1

        # Handle any remaining cells
        if current_batch:
            current_batch.extend(markdown_cells)
            cell_batches.append(current_batch)

        return cell_batches

    def remove_images_from_text(self, text):
        # Remove image tags from markdown text
        import re
        return re.sub(r'!\[.*?\]\(.*?\)', '', text)

    def extract_images_from_output(self, cell):
        # Extract images from output cells and include them as image objects
        for output in cell.outputs:
            if 'data' in output and 'image/png' in output['data']:
                image_data = output['data']['image/png']
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                output['image_object'] = image
                # Remove the image data from the output text
                del output['data']['image/png']
        return cell

    def read_notebook_content(self):
        """
        Extracts text content from code and markdown cells of the loaded notebook, filtering out base64 and outputs.
        """
        if not self.notebook:
            raise ValueError("No notebook loaded. Ensure the directory contains a valid Jupyter notebook.")

        content = []

        # Iterate through the cells in the notebook
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                # Include the source code from code cells, excluding outputs
                if "source" in cell:
                    content.append({
                        "type": "code",
                        "content": cell["source"].strip()
                    })
            elif cell.cell_type == "markdown":
                # Include the content of markdown cells, filtering out base64 strings
                if "source" in cell:
                    markdown_content = cell["source"].strip()
                    filtered_content = "\n".join(
                        line for line in markdown_content.split("\n")
                        if not ("data:image/" in line and "base64," in line)
                    )
                    content.append({
                        "type": "markdown",
                        "content": filtered_content
                    })

        return content
