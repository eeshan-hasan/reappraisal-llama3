from flask import Flask, request, render_template
import subprocess
from script_2 import LLM_output  # Import the function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Call the LLM_output function and get the output
        output = LLM_output(input_text)
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
