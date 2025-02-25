from fadhli_proxy.classes import Proxy
from dotenv import load_dotenv
import json
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import argparse
from pathlib import Path

def ensure_output_dir(dir_path: str) -> Path:
    """Ensure that the output directory exists and return it as a Path object."""
    output = Path(dir_path)
    output.mkdir(exist_ok=True, parents=True)
    return output

def convert_pdf_to_txt(pdf_path: str, output_dir: Path) -> str:
    """
    Converts a PDF file to a text string and saves it in the output directory.
    
    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory where the text file should be saved.
        
    Returns:
        String containing the extracted text.
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()  # May return None if no text is found
                if page_text:
                    text += page_text + "\n"
        
        # Create output path in the output_dir with same name as PDF but .txt extension
        txt_path = output_dir / (Path(pdf_path).stem + '.txt')
        
        # Write the extracted text to the output file
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        
        print(f"Successfully converted PDF: {pdf_path} to {txt_path}")
        return text
    except Exception as e:
        print(f"Error converting PDF to text for {pdf_path}: {e}")
        return ""

def convert_url_to_txt(url: str, output_dir: Path) -> str:
    """
    Fetches the content of a URL, extracts its text, and saves it in the output directory.
    
    Args:
        url: The URL to fetch content from.
        output_dir: Directory where the text file should be saved.
        
    Returns:
        String containing the extracted text.
    """
    try:
        # Extract domain for filename creation
        domain = url.split('//')[-1].split('/')[0]
        txt_path = output_dir / (domain + '.txt')
        
        # Create a session with headers to avoid being blocked
        session = requests.Session()
        session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        })
        
        # Fetch the URL content without using a proxy
        response = session.get(url, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts, styles, and other unwanted elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get the text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text: remove excessive newlines
        text = '\n'.join(line for line in text.splitlines() if line.strip())
        
        # Write to file
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        
        print(f"Successfully converted URL: {url} to {txt_path}")
        return text
    except Exception as e:
        print(f"Error converting URL to text for {url}: {e}")
        return ""

def process_file_or_url(input_path=None, url=None, output_dir: Path = None):
    """
    Process either a PDF file or a URL based on input.
    
    Args:
        input_path: Path to a PDF file to convert.
        url: URL to convert to text.
        output_dir: Directory where the text file should be saved.
    """
    if input_path and os.path.exists(input_path):
        if input_path.lower().endswith('.pdf'):
            print(f"Converting single PDF file: {input_path}")
            convert_pdf_to_txt(input_path, output_dir)
        else:
            print(f"Unsupported file type: {input_path}")
    
    if url:
        print(f"Converting single URL: {url}")
        convert_url_to_txt(url, output_dir)

def process_pdfs_file(pdfs_file: str, output_dir: Path):
    """
    Process multiple PDF files specified in a text file.
    
    Args:
        pdfs_file: Path to the text file containing PDF file paths.
        output_dir: Directory where the text files should be saved.
    """
    try:
        with open(pdfs_file, 'r') as file:
            pdf_paths = file.readlines()
        
        for pdf_path in pdf_paths:
            pdf_path = pdf_path.strip()
            if pdf_path and os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
                print(f"Converting PDF file: {pdf_path}")
                convert_pdf_to_txt(pdf_path, output_dir)
            else:
                print(f"Invalid or unsupported PDF path: {pdf_path}")
    except Exception as e:
        print(f"Error processing PDFs file ({pdfs_file}): {e}")

def process_urls_file(urls_file: str, output_dir: Path):
    """
    Process multiple URLs specified in a text file.
    
    Args:
        urls_file: Path to the text file containing URLs.
        output_dir: Directory where the text files should be saved.
    """
    try:
        with open(urls_file, 'r') as file:
            urls = file.readlines()
        
        for url in urls:
            url = url.strip()
            if url.startswith(('http://', 'https://')):
                print(f"Converting URL: {url}")
                convert_url_to_txt(url, output_dir)
            else:
                print(f"Invalid URL: {url}")
    except Exception as e:
        print(f"Error processing URLs file ({urls_file}): {e}")

def main():
    """Main entry point with command line argument handling."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Convert PDFs or URLs to text files."
    )
    
    # Optional output directory argument
    parser.add_argument(
        "--output-dir",
        help="Directory to save converted text files (default: 'converted')",
        default="converted"
    )
    
    # Single PDF or URL input
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pdf", help="Path to a single PDF file to convert to text")
    group.add_argument("--url", help="A single URL to convert to text")
    
    # File containing multiple paths
    parser.add_argument("--pdfs-file", help="Path to a text file containing PDF file paths")
    parser.add_argument("--urls-file", help="Path to a text file containing URLs")
    
    # Positional argument for auto-detect
    parser.add_argument("input", nargs="?", help="Input file path or URL (will auto-detect type)")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    output_dir = ensure_output_dir(args.output_dir)
    
    # Process explicit arguments first
    if args.pdf:
        process_file_or_url(input_path=args.pdf, output_dir=output_dir)
    if args.url:
        process_file_or_url(url=args.url, output_dir=output_dir)
    if args.pdfs_file:
        process_pdfs_file(args.pdfs_file, output_dir)
    if args.urls_file:
        process_urls_file(args.urls_file, output_dir)
    
    # Auto-detect from positional argument if provided and no other arguments were given
    if not any([args.pdf, args.url, args.pdfs_file, args.urls_file]) and args.input:
        if os.path.exists(args.input):
            process_file_or_url(input_path=args.input, output_dir=output_dir)
        elif args.input.startswith(('http://', 'https://')):
            process_file_or_url(url=args.input, output_dir=output_dir)
        else:
            print(f"Input not recognized: {args.input}")
    elif not any([args.pdf, args.url, args.pdfs_file, args.urls_file, args.input]):
        print("Please provide a PDF file path, URL, or a file containing PDF paths or URLs to convert.")
        parser.print_help()

if __name__ == "__main__":
    main()