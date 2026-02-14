"""
Entry point for the integrated application.
This simply imports the unified Flask app from db_dyn.py and runs it.
"""

from db_dyn import app

if __name__ == '__main__':
    app.run(debug=True, port=5000)