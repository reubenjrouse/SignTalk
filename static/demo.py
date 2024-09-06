from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('femo.html')  # Serve the HTML form

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name')  # Get the 'name' field from the form
    message = request.form.get('message')  # Get the 'message' field from the form
    return f'Hello, {name}! You sent the message: {message}'

if __name__ == '__main__':
    app.run(debug=True)
