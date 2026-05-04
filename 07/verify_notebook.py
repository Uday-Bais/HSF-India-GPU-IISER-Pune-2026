#!/usr/bin/env python3
"""
Notebook verification and analysis script for the CUDA Neural Network Demo
"""

import json
import os
import sys

def analyze_notebook(notebook_path):
    """Analyze the structure and content of the Jupyter notebook"""

    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file '{notebook_path}' not found!")
        return False

    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in notebook file: {e}")
        return False

    print("=== CUDA Neural Network Demo Notebook Analysis ===\n")

    # Basic statistics
    total_cells = len(notebook['cells'])
    markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')

    print(f"Total cells: {total_cells}")
    print(f"Markdown cells: {markdown_cells}")
    print(f"Code cells: {code_cells}")
    print(f"Notebook format: {notebook['nbformat']}.{notebook['nbformat_minor']}")

    # Check kernel specification
    if 'kernelspec' in notebook['metadata']:
        kernel = notebook['metadata']['kernelspec']
        print(f"Kernel: {kernel['display_name']} ({kernel['name']})")

    print("\n=== Content Structure ===\n")

    # Analyze content structure
    section_count = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if source.startswith('## '):
                section_count += 1
                title = source.split('\n')[0].replace('## ', '')
                print(f"Section {section_count}: {title}")

    print("\n=== Code Languages Detected ===\n")

    # Detect programming languages in code cells
    languages = set()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip():
                # Simple heuristics to detect language
                if '#include' in source or '__device__' in source or '__global__' in source:
                    languages.add('CUDA C++')
                elif 'import' in source and 'matplotlib' in source:
                    languages.add('Python')
                elif 'mkdir' in source or './json_generator' in source:
                    languages.add('Bash')
                elif 'cmake_minimum_required' in source:
                    languages.add('CMake')
                elif '{' in source and '"num_input"' in source:
                    languages.add('JSON')
                else:
                    languages.add('C++')

    for lang in sorted(languages):
        print(f"- {lang}")

    print("\n=== Key Features Covered ===\n")

    # Check for key topics covered
    topics = {
        'Neural Network Architecture': False,
        'CUDA Kernels': False,
        'Memory Management': False,
        'JSON Configuration': False,
        'Template Programming': False,
        'Allen Framework': False,
        'Performance Optimization': False,
        'Build System': False
    }

    full_content = ''
    for cell in notebook['cells']:
        full_content += ''.join(cell['source']).lower()

    if 'neural network' in full_content and 'architecture' in full_content:
        topics['Neural Network Architecture'] = True
    if '__global__' in full_content or '__device__' in full_content:
        topics['CUDA Kernels'] = True
    if 'cudamalloc' in full_content or 'memory' in full_content:
        topics['Memory Management'] = True
    if 'json' in full_content and 'model' in full_content:
        topics['JSON Configuration'] = True
    if 'template' in full_content:
        topics['Template Programming'] = True
    if 'allen' in full_content:
        topics['Allen Framework'] = True
    if 'optimization' in full_content or 'performance' in full_content:
        topics['Performance Optimization'] = True
    if 'cmake' in full_content or 'build' in full_content:
        topics['Build System'] = True

    for topic, covered in topics.items():
        status = "✓" if covered else "✗"
        print(f"{status} {topic}")

    print("\n=== Verification Complete ===\n")

    coverage = sum(topics.values()) / len(topics) * 100
    print(f"Topic coverage: {coverage:.1f}%")

    if coverage >= 80:
        print("✓ Excellent comprehensive coverage!")
    elif coverage >= 60:
        print("✓ Good coverage of key topics")
    else:
        print("⚠ Some key topics may be missing")

    return True

def extract_code_files(notebook_path, output_dir="extracted_code"):
    """Extract code cells to separate files"""

    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n=== Extracting Code Files to {output_dir}/ ===\n")

    file_count = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip():
                # Determine file extension based on content
                if '__device__' in source or '__global__' in source:
                    ext = 'cu'
                elif '#include <iostream>' in source and 'int main' in source:
                    ext = 'cpp'
                elif 'cmake_minimum_required' in source:
                    ext = 'cmake'
                elif source.strip().startswith('{'):
                    ext = 'json'
                elif source.strip().startswith('#') or 'mkdir' in source:
                    ext = 'sh'
                else:
                    ext = 'txt'

                filename = f"cell_{i:02d}.{ext}"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'w') as f:
                    f.write(source)

                print(f"Created: {filename}")
                file_count += 1

    print(f"\nExtracted {file_count} code files")

if __name__ == "__main__":
    notebook_path = "cuda_neural_network_demo_complete.ipynb"

    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    success = analyze_notebook(notebook_path)

    if success:
        extract_code_files(notebook_path)

    print("\nDone!")
