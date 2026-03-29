from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from werkzeug.security import generate_password_hash, check_password_hash

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=api_key)

# Company tiers
company_tiers = {
    "Tier 1": ["Google", "Microsoft", "Amazon", "Adobe", "Goldman Sachs"],
    "Tier 2": ["Oracle", "IBM", "Accenture", "Capgemini", "Deloitte"],
    "Tier 3": ["TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra"],
    "Tier 4": ["Startups", "Mid-size IT Firms", "Local Companies"],
    "Tier 5": ["Internships / Off-campus roles"]
}

company_logos = {
    "Google": "logos/google.png",
    "Microsoft": "logos/Microsoft.png",
    "Amazon": "logos/amazon.png",
    "Adobe": "logos/adobe.png",
    "Goldman Sachs": "logos/goldman.png",
    "Oracle": "logos/oracle.png",
    "IBM": "logos/ibm.png",
    "Accenture": "logos/Accenture.png",
    "Capgemini": "logos/capgemini.png",
    "Deloitte": "logos/deloitte.png",
    "TCS": "logos/tcs.png",
    "Infosys": "logos/infosys.png",
    "Wipro": "logos/Wipro.png",
    "HCL": "logos/HCL.png",
    "Tech Mahindra": "logos/mahindra.png",
    "Startups": "logos/startup.png",
    "Mid-size IT Firms": "logos/it.png",
    "Local Companies": "logos/local.png",
    "Internships / Off-campus roles": "logos/internship.png"
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

# ---------------- DATABASE ---------------- #

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    result = db.Column(db.String(50))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- ML MODEL ---------------- #

data = pd.read_csv("placementdata.csv")

# Convert target
data["PlacementStatus"] = data["PlacementStatus"].map({
    "Placed": 1,
    "NotPlaced": 0
})

# Convert categorical columns
data = data.replace({
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
})

# Drop unnecessary column
data = data.drop("StudentID", axis=1)

# Keep only numeric columns
data = data.select_dtypes(include=['int64', 'float64'])

# Features & Target
X = data.drop("PlacementStatus", axis=1)
y = data["PlacementStatus"]

# ---------------- MODEL TRAINING ---------------- #
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
logistic_acc = model.score(X_test, y_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = dt.score(X_test, y_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

def get_ai_suggestion(vals, confidence):
    prompt = f"""
    A student profile:

    CGPA: {vals[0]}
    Internships: {vals[1]}
    Projects: {vals[2]}
    Workshops: {vals[3]}
    Aptitude Score: {vals[4]}
    Soft Skills: {vals[5]}
    10th Marks: {vals[6]}
    12th Marks: {vals[7]}

    Placement Confidence: {confidence}%

    Give short, practical suggestions on:
    - What to improve
    - Skills to focus on
    - How to increase placement chances
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

models = {
    "Logistic": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

scores = []

for name, m in models.items():
    m.fit(X, y)
    scores.append(m.score(X, y))

plt.figure(figsize=(6,4))
plt.bar(models.keys(), scores)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("static/graph.png")
plt.close()


# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return redirect("/login")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        hashed_password = generate_password_hash(request.form["password"])
        user = User(username=request.form["username"], password=hashed_password)
        db.session.add(user)
        db.session.commit()
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            return redirect("/dashboard")
    return render_template("login.html")

@app.route("/dashboard", methods=["GET","POST"])
@login_required
def dashboard():
    result = None
    confidence = None
    recommended_companies = []
    company_result = None
    company = "General" 
    ai_suggestion = None
    if request.method == "POST":
        vals = [
            float(request.form["cgpa"]),
            float(request.form["intern"]),
            float(request.form["proj"]),
            float(request.form["work"]),
            float(request.form["apt"]),
            float(request.form["soft"]),
            float(request.form["10th"]),
            float(request.form["12th"])
        ]
        proba = model.predict_proba([vals])[0][1]   # probability of placement
        pred = 1 if proba >= 0.5 else 0

        result = "Placed" if pred == 1 else "Not Placed"
        confidence = round(proba * 100, 2)

        try:
            ai_suggestion = get_ai_suggestion(vals, confidence)
        except:
            ai_suggestion = "Improve projects, aptitude, and communication skills."
        
        # ---------------- COMPANY BASED ADJUSTMENT ---------------- #

        company = request.form.get("company")

        company_weight = {
            "Google": 0.9,
            "Microsoft": 0.85,
            "Amazon": 0.8,
            "Oracle": 0.7,
            "TCS": 0.6,
            "Infosys": 0.6,
            "Wipro": 0.55,
            "General": 0.7
        }

        weight = company_weight.get(company, 0.7)

        adjusted_confidence = confidence * weight
        adjusted_confidence = round(adjusted_confidence, 2)

        if adjusted_confidence >= 60:
            company_result = f"High chances for {company}"
        elif adjusted_confidence >= 40:
            company_result = f"Moderate chances for {company}"
        else:
            company_result = f"Low chances for {company}"
        
        if confidence >= 80:
            tier = ["Tier 1", "Tier 2"]
        elif confidence >= 60:
            tier = ["Tier 2", "Tier 3"]
        elif confidence >= 40:
            tier = ["Tier 3", "Tier 4"]
        else:
            tier = ["Tier 4", "Tier 5"]

        recommended_companies = []
        for t in tier:
            recommended_companies.extend(company_tiers[t])

        db.session.add(Prediction(user_id=current_user.id, result=result))
        db.session.commit()

    history = Prediction.query.filter_by(user_id=current_user.id).all()
    
    return render_template(
        "dashboard.html",
        result=result,
        history=history,
        confidence=confidence,
        companies=recommended_companies,
        company_result=company_result,
        selected_company=company,
        company_logos=company_logos,
        ai_suggestion=ai_suggestion,
        logistic_acc=round(logistic_acc, 2),
        dt_acc=round(dt_acc, 2),
        rf_acc=round(rf_acc, 2)
    )
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

@app.route("/terms")
def terms():
    return render_template("terms.html")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)