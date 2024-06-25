import sys

if len(sys.argv) > 1:
    input_text = sys.argv[1]
    output_text = input_text[::-1]
    print(output_text)
else:
    print("No input provided.")
