from flask import Flask, render_template, request, make_response

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default values for the plot
    x_new = [1, 2, 3, 4, 5]
    y_new = [1, 4, 9, 16, 25]

    # Check if the form has been submitted
    if request.method == 'POST':
        basil_rate = request.form['basilRate']
        resp = make_response(render_template('index.html', x_new=x_new, y_new=y_new))
        resp.set_cookie('basilRate', basil_rate)
        return resp

    # Attempt to read the basilRate from the cookie
    basil_rate = request.cookies.get('basilRate', 'Not set')

    return render_template('index.html', x_new=x_new, y_new=y_new, basil_rate=basil_rate)

if __name__ == '__main__':
    app.run(debug=True)