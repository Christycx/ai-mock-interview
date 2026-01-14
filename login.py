from flask import Flask, request, jsonify, render_template
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# -------- MYSQL CONFIG --------
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'interview@1234'
app.config['MYSQL_DB'] = 'mockinterview'

mysql = MySQL(app)

# -------- LOAD LOGIN PAGE --------
@app.route('/')
def index():
    return render_template('login.html')

# -------- SIGNUP --------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data['name']
    email = data['email']
    password = data['password']

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM login WHERE email=%s", (email,))
    if cur.fetchone():
        return jsonify({"error": "User already exists"}), 409

    cur.execute(
        "INSERT INTO login (name, email, password) VALUES (%s, %s, %s)",
        (name, email, hashed_pw)
    )
    mysql.connection.commit()
    cur.close()

    return jsonify({"message": "Signup successful"}), 201

# -------- LOGIN --------
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data['email']
    password = data['password']

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT student_id, name, email, password FROM login WHERE email=%s",
        (email,)
    )
    user = cur.fetchone()
    cur.close()

    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    student_id, name, email, hashed_pw = user

    if not bcrypt.check_password_hash(hashed_pw, password):
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({
        "message": "Login successful",
        "user": {
            "student_id": student_id,
            "name": name,
            "email": email
        }
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)