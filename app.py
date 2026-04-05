from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask import Response
import json
from pathlib import Path
from services.student_service import StudentService
from services.hero_service import HeroService
from services.dialogue_service import DialogueService
from services.session_store import SessionStore
from schemas.dto import CreateStudentRequest, CreateSingleSessionRequest, CreateMultiSessionRequest

# 初始化服务
app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

student_service = StudentService(PROJECT_ROOT)
hero_service = HeroService()
dialogue_service = DialogueService(PROJECT_ROOT)
session_store = SessionStore(PROJECT_ROOT)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/students")
def list_students():
    students = student_service.list_students()
    return render_template("students.html", students=students)


@app.route("/students/create", methods=["GET", "POST"])
def create_student():
    if request.method == "POST":
        student_data = request.form
        payload = {
            "name": student_data["name"],
            "grade": student_data["grade"],
            "major": student_data["major"],
            "identity": student_data["identity"],
            "self_view": student_data["self_view"],
            "personality_traits": student_data["personality_traits"].split("\n"),
            "speaking_style": student_data["speaking_style"].split("\n"),
            "interests": student_data["interests"].split("\n"),
            "strengths": student_data["strengths"].split("\n"),
            "weaknesses": student_data["weaknesses"].split("\n"),
            "current_stage": student_data["current_stage"],
            "learning_goals": student_data["learning_goals"].split("\n"),
            "current_confusions": student_data["current_confusions"].split("\n"),
            "concerns": student_data["concerns"].split("\n"),
            "values": student_data["values"].split("\n"),
            "expectations": student_data["expectations"],
            "common_questions": student_data["common_questions"].split("\n"),
            "response_preference": student_data["response_preference"].split("\n"),
            "opening_intro": student_data["opening_intro"],
            "opening_problem": student_data["opening_problem"],
            "opening_expectation": student_data["opening_expectation"],
        }

        student_service.create_student(payload)
        return redirect(url_for("list_students"))

    return render_template("create_student.html")


@app.route("/heroes")
def list_heroes():
    heroes = hero_service.list_heroes()
    return render_template("heroes.html", heroes=heroes)


@app.route("/sessions")
def list_sessions():
    sessions = session_store.list_sessions()
    return render_template("sessions.html", sessions=sessions)


@app.route("/sessions/single/create", methods=["POST"])
def create_single_session():
    payload = request.json
    student_doc_id = payload["student_doc_id"]
    hero_doc_id = payload["hero_doc_id"]
    session = dialogue_service.create_single_session(student_doc_id, hero_doc_id)
    return jsonify(session)


@app.route("/sessions/multi/create", methods=["POST"])
def create_multi_session():
    payload = request.json
    student_doc_id = payload["student_doc_id"]
    hero_doc_ids = payload["hero_doc_ids"]
    session = dialogue_service.create_multi_session(student_doc_id, hero_doc_ids)
    return jsonify(session)


@app.route("/sessions/next-round", methods=["POST"])
def next_round():
    payload = request.json
    session_id = payload["session_id"]
    result = dialogue_service.run_next_round(session_id)
    return jsonify(result)


@app.route("/sessions/<session_id>")
def get_session(session_id):
    session = session_store.load_session(session_id)
    return render_template("session_detail.html", session=session)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)