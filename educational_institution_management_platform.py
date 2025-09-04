"""
Educational Institution Management Platform - Complete AI-Powered Education System
$15B+ Value Potential - Student Performance, Administrative Automation & Resource Optimization
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import pandas as pd
import numpy as np
from openai import OpenAI
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "education-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///education.db")

db.init_app(app)

# Education Enums
class InstitutionType(Enum):
    ELEMENTARY = "elementary"
    MIDDLE_SCHOOL = "middle_school"
    HIGH_SCHOOL = "high_school"
    COLLEGE = "college"
    UNIVERSITY = "university"
    VOCATIONAL = "vocational"

class StudentStatus(Enum):
    ACTIVE = "active"
    GRADUATED = "graduated"
    WITHDRAWN = "withdrawn"
    SUSPENDED = "suspended"

class GradeLevel(Enum):
    KINDERGARTEN = "k"
    FIRST = "1st"
    SECOND = "2nd"
    THIRD = "3rd"
    FOURTH = "4th"
    FIFTH = "5th"
    SIXTH = "6th"
    SEVENTH = "7th"
    EIGHTH = "8th"
    NINTH = "9th"
    TENTH = "10th"
    ELEVENTH = "11th"
    TWELFTH = "12th"

# Data Models
class EducationalInstitution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    institution_id = db.Column(db.String(100), unique=True, nullable=False)
    institution_name = db.Column(db.String(200), nullable=False)
    institution_type = db.Column(db.Enum(InstitutionType), nullable=False)
    
    # Institution Details
    location = db.Column(db.String(200))
    student_capacity = db.Column(db.Integer, default=500)
    current_enrollment = db.Column(db.Integer, default=0)
    faculty_count = db.Column(db.Integer, default=25)
    
    # Academic Configuration
    academic_year_start = db.Column(db.Date)
    academic_year_end = db.Column(db.Date)
    grading_system = db.Column(db.String(50), default='letter')  # letter, numeric, pass_fail
    
    # Performance Metrics
    graduation_rate = db.Column(db.Float, default=85.0)  # percentage
    student_satisfaction = db.Column(db.Float, default=8.0)  # 1-10 scale
    teacher_retention_rate = db.Column(db.Float, default=90.0)  # percentage
    
    # Financial Information
    annual_budget = db.Column(db.Float, default=0.0)
    cost_per_student = db.Column(db.Float, default=8000.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(100), unique=True, nullable=False)
    institution_id = db.Column(db.String(100), db.ForeignKey('educational_institution.institution_id'), nullable=False)
    
    # Personal Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    date_of_birth = db.Column(db.Date)
    grade_level = db.Column(db.String(10))
    status = db.Column(db.Enum(StudentStatus), default=StudentStatus.ACTIVE)
    
    # Contact Information
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    address = db.Column(db.Text)
    
    # Guardian Information
    guardian_name = db.Column(db.String(200))
    guardian_email = db.Column(db.String(200))
    guardian_phone = db.Column(db.String(50))
    
    # Academic Performance
    current_gpa = db.Column(db.Float, default=0.0)
    cumulative_gpa = db.Column(db.Float, default=0.0)
    class_rank = db.Column(db.Integer)
    
    # Attendance
    attendance_rate = db.Column(db.Float, default=95.0)  # percentage
    absences_total = db.Column(db.Integer, default=0)
    tardiness_count = db.Column(db.Integer, default=0)
    
    # Behavioral Records
    disciplinary_actions = db.Column(db.JSON)  # List of disciplinary records
    counseling_sessions = db.Column(db.JSON)  # List of counseling sessions
    
    # Special Needs
    special_needs = db.Column(db.JSON)  # IEP, 504 plans, etc.
    accommodations = db.Column(db.JSON)
    
    # AI Predictions
    graduation_probability = db.Column(db.Float, default=85.0)  # percentage
    college_readiness_score = db.Column(db.Float, default=75.0)  # 0-100
    risk_level = db.Column(db.String(20), default='low')  # low, medium, high
    recommended_interventions = db.Column(db.JSON)
    
    # Extracurricular
    clubs_activities = db.Column(db.JSON)
    sports_participation = db.Column(db.JSON)
    volunteer_hours = db.Column(db.Integer, default=0)
    
    enrollment_date = db.Column(db.Date)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.String(100), unique=True, nullable=False)
    institution_id = db.Column(db.String(100), db.ForeignKey('educational_institution.institution_id'), nullable=False)
    
    # Course Information
    course_name = db.Column(db.String(200), nullable=False)
    course_code = db.Column(db.String(50))
    department = db.Column(db.String(100))
    credit_hours = db.Column(db.Integer, default=3)
    
    # Course Details
    description = db.Column(db.Text)
    prerequisites = db.Column(db.JSON)  # List of required courses
    learning_objectives = db.Column(db.JSON)
    grade_levels = db.Column(db.JSON)  # Applicable grade levels
    
    # Instructor Information
    instructor_id = db.Column(db.String(100))
    instructor_name = db.Column(db.String(200))
    
    # Enrollment
    max_enrollment = db.Column(db.Integer, default=30)
    current_enrollment = db.Column(db.Integer, default=0)
    waitlist_count = db.Column(db.Integer, default=0)
    
    # Performance Metrics
    average_grade = db.Column(db.Float, default=0.0)
    pass_rate = db.Column(db.Float, default=85.0)  # percentage
    student_satisfaction = db.Column(db.Float, default=8.0)  # 1-10 scale
    
    # Schedule
    meeting_days = db.Column(db.JSON)  # Days of week
    meeting_times = db.Column(db.JSON)  # Start/end times
    classroom = db.Column(db.String(100))
    
    # AI Analysis
    difficulty_level = db.Column(db.Float, default=5.0)  # 1-10 scale
    recommended_study_hours = db.Column(db.Float, default=6.0)  # per week
    success_predictors = db.Column(db.JSON)
    
    semester = db.Column(db.String(20))
    academic_year = db.Column(db.String(10))

class Grade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    grade_id = db.Column(db.String(100), unique=True, nullable=False)
    student_id = db.Column(db.String(100), db.ForeignKey('student.student_id'), nullable=False)
    course_id = db.Column(db.String(100), db.ForeignKey('course.course_id'), nullable=False)
    
    # Assignment/Test Information
    assignment_name = db.Column(db.String(200), nullable=False)
    assignment_type = db.Column(db.String(50))  # homework, quiz, test, project, final
    
    # Grading
    points_earned = db.Column(db.Float, nullable=False)
    points_possible = db.Column(db.Float, nullable=False)
    percentage = db.Column(db.Float)
    letter_grade = db.Column(db.String(5))
    
    # Timing
    assignment_date = db.Column(db.Date)
    due_date = db.Column(db.Date)
    submitted_date = db.Column(db.Date)
    
    # Status
    is_late = db.Column(db.Boolean, default=False)
    is_makeup = db.Column(db.Boolean, default=False)
    is_extra_credit = db.Column(db.Boolean, default=False)
    
    # Feedback
    instructor_comments = db.Column(db.Text)
    rubric_scores = db.Column(db.JSON)  # Detailed rubric scoring
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Teacher(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.String(100), unique=True, nullable=False)
    institution_id = db.Column(db.String(100), db.ForeignKey('educational_institution.institution_id'), nullable=False)
    
    # Personal Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    
    # Professional Information
    department = db.Column(db.String(100))
    position = db.Column(db.String(100))
    subjects_taught = db.Column(db.JSON)
    grade_levels_taught = db.Column(db.JSON)
    
    # Qualifications
    education_level = db.Column(db.String(50))
    certifications = db.Column(db.JSON)
    years_experience = db.Column(db.Integer, default=0)
    
    # Performance Metrics
    student_evaluation_score = db.Column(db.Float, default=8.0)  # 1-10 scale
    class_average_performance = db.Column(db.Float, default=80.0)  # percentage
    professional_development_hours = db.Column(db.Integer, default=0)
    
    # Workload
    courses_taught = db.Column(db.JSON)  # List of course IDs
    total_students = db.Column(db.Integer, default=0)
    prep_periods = db.Column(db.Integer, default=1)
    
    # AI Analysis
    effectiveness_score = db.Column(db.Float, default=85.0)  # 0-100
    improvement_areas = db.Column(db.JSON)
    recommended_training = db.Column(db.JSON)
    
    hire_date = db.Column(db.Date)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class ResourceAllocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    allocation_id = db.Column(db.String(100), unique=True, nullable=False)
    institution_id = db.Column(db.String(100), db.ForeignKey('educational_institution.institution_id'), nullable=False)
    
    # Resource Details
    resource_type = db.Column(db.String(100), nullable=False)  # classroom, equipment, budget, staff
    resource_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # Allocation Information
    allocated_to = db.Column(db.String(100))  # department, course, or specific use
    allocation_date = db.Column(db.Date, default=datetime.utcnow().date)
    
    # Capacity and Utilization
    total_capacity = db.Column(db.Integer, default=1)
    current_utilization = db.Column(db.Integer, default=0)
    utilization_percentage = db.Column(db.Float, default=0.0)
    
    # Financial
    cost = db.Column(db.Float, default=0.0)
    annual_maintenance_cost = db.Column(db.Float, default=0.0)
    
    # Performance
    condition_rating = db.Column(db.Float, default=8.0)  # 1-10 scale
    effectiveness_rating = db.Column(db.Float, default=8.0)  # 1-10 scale
    
    # AI Optimization
    optimization_score = db.Column(db.Float, default=75.0)  # 0-100
    recommended_actions = db.Column(db.JSON)
    
    last_maintenance = db.Column(db.Date)
    next_review_date = db.Column(db.Date)

# Educational Intelligence Engine
class EducationalIntelligenceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def predict_student_performance(self, institution_id: str) -> Dict[str, Any]:
        """AI-powered student performance prediction and intervention system"""
        
        students = Student.query.filter_by(institution_id=institution_id, status=StudentStatus.ACTIVE).all()
        
        if not students:
            return {'error': 'No active students found'}
        
        performance_predictions = []
        intervention_recommendations = []
        
        for student in students:
            prediction = self._predict_individual_student_performance(student)
            performance_predictions.append(prediction)
            
            if prediction['risk_level'] in ['medium', 'high']:
                interventions = self._recommend_student_interventions(student, prediction)
                intervention_recommendations.extend(interventions)
        
        # Aggregate analysis
        aggregate_analysis = self._analyze_student_population(students, performance_predictions)
        
        return {
            'institution_id': institution_id,
            'students_analyzed': len(students),
            'performance_predictions': performance_predictions,
            'intervention_recommendations': intervention_recommendations,
            'population_analysis': aggregate_analysis,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _predict_individual_student_performance(self, student: Student) -> Dict[str, Any]:
        """Predict individual student academic performance and graduation probability"""
        
        # Get student's recent grades
        recent_grades = Grade.query.filter_by(student_id=student.student_id)\
                                  .order_by(Grade.assignment_date.desc())\
                                  .limit(20).all()
        
        # Calculate performance indicators
        performance_indicators = self._calculate_performance_indicators(student, recent_grades)
        
        # Risk assessment
        risk_factors = self._assess_student_risk_factors(student, performance_indicators)
        
        # AI-powered prediction
        predictions = self._generate_performance_predictions(student, performance_indicators, risk_factors)
        
        return {
            'student_id': student.student_id,
            'student_name': f"{student.first_name} {student.last_name}",
            'current_gpa': student.current_gpa,
            'performance_indicators': performance_indicators,
            'risk_factors': risk_factors,
            'predictions': predictions,
            'risk_level': self._calculate_overall_risk_level(risk_factors)
        }
    
    def _calculate_performance_indicators(self, student: Student, recent_grades: List[Grade]) -> Dict[str, Any]:
        """Calculate key performance indicators for student"""
        
        if not recent_grades:
            return {'error': 'No recent grades available'}
        
        # Grade trend analysis
        grade_percentages = [grade.percentage for grade in recent_grades if grade.percentage]
        if grade_percentages:
            recent_average = np.mean(grade_percentages[-5:]) if len(grade_percentages) >= 5 else np.mean(grade_percentages)
            older_average = np.mean(grade_percentages[:-5]) if len(grade_percentages) > 5 else recent_average
            grade_trend = 'improving' if recent_average > older_average + 2 else ('declining' if recent_average < older_average - 2 else 'stable')
        else:
            grade_trend = 'unknown'
        
        # Assignment completion rate
        total_assignments = len(recent_grades)
        on_time_submissions = len([g for g in recent_grades if not g.is_late])
        completion_rate = (on_time_submissions / total_assignments * 100) if total_assignments > 0 else 100
        
        # Performance consistency
        grade_variance = np.var(grade_percentages) if grade_percentages else 0
        consistency_score = max(0, 100 - grade_variance)  # Lower variance = higher consistency
        
        return {
            'current_gpa': student.current_gpa,
            'grade_trend': grade_trend,
            'recent_average': recent_average if grade_percentages else 0,
            'assignment_completion_rate': completion_rate,
            'performance_consistency': consistency_score,
            'attendance_rate': student.attendance_rate
        }
    
    def _assess_student_risk_factors(self, student: Student, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Assess various risk factors for student success"""
        
        risk_factors = {
            'academic_risk': 0,
            'attendance_risk': 0,
            'behavioral_risk': 0,
            'engagement_risk': 0
        }
        
        # Academic risk
        if student.current_gpa < 2.0:
            risk_factors['academic_risk'] = 80
        elif student.current_gpa < 2.5:
            risk_factors['academic_risk'] = 60
        elif student.current_gpa < 3.0:
            risk_factors['academic_risk'] = 30
        else:
            risk_factors['academic_risk'] = 10
        
        # Attendance risk
        if student.attendance_rate < 80:
            risk_factors['attendance_risk'] = 90
        elif student.attendance_rate < 90:
            risk_factors['attendance_risk'] = 50
        elif student.attendance_rate < 95:
            risk_factors['attendance_risk'] = 20
        else:
            risk_factors['attendance_risk'] = 5
        
        # Behavioral risk
        disciplinary_actions = student.disciplinary_actions or []
        if len(disciplinary_actions) > 3:
            risk_factors['behavioral_risk'] = 70
        elif len(disciplinary_actions) > 1:
            risk_factors['behavioral_risk'] = 40
        elif len(disciplinary_actions) > 0:
            risk_factors['behavioral_risk'] = 20
        else:
            risk_factors['behavioral_risk'] = 5
        
        # Engagement risk (based on extracurricular participation)
        activities = (student.clubs_activities or []) + (student.sports_participation or [])
        if len(activities) == 0:
            risk_factors['engagement_risk'] = 50
        elif len(activities) == 1:
            risk_factors['engagement_risk'] = 25
        else:
            risk_factors['engagement_risk'] = 10
        
        return risk_factors
    
    def _generate_performance_predictions(self, student: Student, performance: Dict, risk_factors: Dict) -> Dict[str, Any]:
        """Generate AI-powered performance predictions"""
        
        # Calculate weighted risk score
        total_risk = (
            risk_factors['academic_risk'] * 0.4 +
            risk_factors['attendance_risk'] * 0.3 +
            risk_factors['behavioral_risk'] * 0.2 +
            risk_factors['engagement_risk'] * 0.1
        )
        
        # Graduation probability
        graduation_probability = max(20, 100 - total_risk)
        
        # GPA prediction for next semester
        current_gpa = student.current_gpa
        if performance['grade_trend'] == 'improving':
            predicted_gpa = min(4.0, current_gpa + 0.2)
        elif performance['grade_trend'] == 'declining':
            predicted_gpa = max(0.0, current_gpa - 0.3)
        else:
            predicted_gpa = current_gpa
        
        # College readiness (for high school students)
        college_readiness = min(100, graduation_probability + (current_gpa / 4.0 * 25))
        
        return {
            'graduation_probability': graduation_probability,
            'predicted_next_semester_gpa': predicted_gpa,
            'college_readiness_score': college_readiness,
            'total_risk_score': total_risk,
            'predicted_outcomes': self._predict_specific_outcomes(student, total_risk)
        }
    
    def _predict_specific_outcomes(self, student: Student, risk_score: float) -> Dict[str, Any]:
        """Predict specific educational outcomes"""
        
        outcomes = {}
        
        if risk_score > 70:
            outcomes['primary_concern'] = 'High dropout risk'
            outcomes['recommended_timeline'] = 'Immediate intervention needed'
        elif risk_score > 50:
            outcomes['primary_concern'] = 'Academic struggles likely'
            outcomes['recommended_timeline'] = 'Intervention within 2 weeks'
        elif risk_score > 30:
            outcomes['primary_concern'] = 'May need additional support'
            outcomes['recommended_timeline'] = 'Monitor and provide support'
        else:
            outcomes['primary_concern'] = 'On track for success'
            outcomes['recommended_timeline'] = 'Continue current approach'
        
        return outcomes
    
    def _calculate_overall_risk_level(self, risk_factors: Dict[str, Any]) -> str:
        """Calculate overall risk level for student"""
        
        avg_risk = np.mean(list(risk_factors.values()))
        
        if avg_risk > 60:
            return 'high'
        elif avg_risk > 35:
            return 'medium'
        else:
            return 'low'
    
    def _recommend_student_interventions(self, student: Student, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend specific interventions for at-risk students"""
        
        interventions = []
        risk_factors = prediction['risk_factors']
        
        # Academic interventions
        if risk_factors['academic_risk'] > 50:
            interventions.append({
                'type': 'academic_support',
                'intervention': 'Assign academic tutor or mentor',
                'urgency': 'high' if risk_factors['academic_risk'] > 70 else 'medium',
                'expected_impact': 'Improve GPA by 0.3-0.5 points'
            })
        
        # Attendance interventions
        if risk_factors['attendance_risk'] > 40:
            interventions.append({
                'type': 'attendance_support',
                'intervention': 'Meet with student and parents to address attendance barriers',
                'urgency': 'high' if risk_factors['attendance_risk'] > 70 else 'medium',
                'expected_impact': 'Increase attendance rate by 10-15%'
            })
        
        # Behavioral interventions
        if risk_factors['behavioral_risk'] > 30:
            interventions.append({
                'type': 'behavioral_support',
                'intervention': 'Counseling and behavioral intervention plan',
                'urgency': 'medium',
                'expected_impact': 'Reduce disciplinary incidents by 50%'
            })
        
        # Engagement interventions
        if risk_factors['engagement_risk'] > 30:
            interventions.append({
                'type': 'engagement_support',
                'intervention': 'Encourage participation in extracurricular activities',
                'urgency': 'low',
                'expected_impact': 'Improve school connection and motivation'
            })
        
        return interventions
    
    def _analyze_student_population(self, students: List[Student], predictions: List[Dict]) -> Dict[str, Any]:
        """Analyze overall student population trends"""
        
        # Risk distribution
        risk_levels = [p['risk_level'] for p in predictions]
        risk_distribution = {
            'high_risk': risk_levels.count('high'),
            'medium_risk': risk_levels.count('medium'),
            'low_risk': risk_levels.count('low')
        }
        
        # Performance metrics
        avg_gpa = np.mean([s.current_gpa for s in students if s.current_gpa])
        avg_attendance = np.mean([s.attendance_rate for s in students])
        avg_graduation_prob = np.mean([p['predictions']['graduation_probability'] for p in predictions])
        
        # Trends identification
        declining_students = len([p for p in predictions if p['performance_indicators']['grade_trend'] == 'declining'])
        improving_students = len([p for p in predictions if p['performance_indicators']['grade_trend'] == 'improving'])
        
        return {
            'total_students': len(students),
            'risk_distribution': risk_distribution,
            'average_gpa': avg_gpa,
            'average_attendance_rate': avg_attendance,
            'average_graduation_probability': avg_graduation_prob,
            'students_declining': declining_students,
            'students_improving': improving_students,
            'high_priority_interventions_needed': risk_distribution['high_risk'] + risk_distribution['medium_risk']
        }
    
    def optimize_resource_allocation(self, institution_id: str) -> Dict[str, Any]:
        """AI-powered resource allocation optimization"""
        
        # Get all resources
        resources = ResourceAllocation.query.filter_by(institution_id=institution_id).all()
        
        if not resources:
            return {'error': 'No resources found'}
        
        # Analyze current utilization
        utilization_analysis = self._analyze_resource_utilization(resources)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_resource_optimizations(resources)
        
        # Generate recommendations
        recommendations = self._generate_resource_recommendations(utilization_analysis, optimization_opportunities)
        
        return {
            'institution_id': institution_id,
            'resources_analyzed': len(resources),
            'utilization_analysis': utilization_analysis,
            'optimization_opportunities': optimization_opportunities,
            'recommendations': recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_resource_utilization(self, resources: List[ResourceAllocation]) -> Dict[str, Any]:
        """Analyze current resource utilization patterns"""
        
        utilization_by_type = {}
        
        for resource in resources:
            resource_type = resource.resource_type
            if resource_type not in utilization_by_type:
                utilization_by_type[resource_type] = {
                    'total_resources': 0,
                    'total_capacity': 0,
                    'total_utilization': 0,
                    'resources': []
                }
            
            utilization_by_type[resource_type]['total_resources'] += 1
            utilization_by_type[resource_type]['total_capacity'] += resource.total_capacity
            utilization_by_type[resource_type]['total_utilization'] += resource.current_utilization
            utilization_by_type[resource_type]['resources'].append({
                'name': resource.resource_name,
                'utilization_percentage': resource.utilization_percentage,
                'condition_rating': resource.condition_rating
            })
        
        # Calculate utilization rates
        for resource_type, data in utilization_by_type.items():
            if data['total_capacity'] > 0:
                data['overall_utilization_rate'] = (data['total_utilization'] / data['total_capacity']) * 100
            else:
                data['overall_utilization_rate'] = 0
        
        return utilization_by_type
    
    def _identify_resource_optimizations(self, resources: List[ResourceAllocation]) -> List[Dict[str, Any]]:
        """Identify resource optimization opportunities"""
        
        opportunities = []
        
        for resource in resources:
            # Underutilized resources
            if resource.utilization_percentage < 50:
                opportunities.append({
                    'type': 'underutilization',
                    'resource_name': resource.resource_name,
                    'resource_type': resource.resource_type,
                    'current_utilization': resource.utilization_percentage,
                    'opportunity': 'Increase utilization or reallocate',
                    'potential_savings': resource.cost * 0.3
                })
            
            # Overutilized resources
            elif resource.utilization_percentage > 95:
                opportunities.append({
                    'type': 'overutilization',
                    'resource_name': resource.resource_name,
                    'resource_type': resource.resource_type,
                    'current_utilization': resource.utilization_percentage,
                    'opportunity': 'Expand capacity or redistribute load',
                    'investment_needed': resource.cost * 0.5
                })
            
            # Poor condition resources
            if resource.condition_rating < 6:
                opportunities.append({
                    'type': 'maintenance_needed',
                    'resource_name': resource.resource_name,
                    'resource_type': resource.resource_type,
                    'condition_rating': resource.condition_rating,
                    'opportunity': 'Repair or replace to improve effectiveness',
                    'estimated_cost': resource.annual_maintenance_cost * 2
                })
        
        return opportunities
    
    def _generate_resource_recommendations(self, utilization: Dict, opportunities: List[Dict]) -> List[Dict[str, Any]]:
        """Generate specific resource optimization recommendations"""
        
        recommendations = []
        
        # Utilization-based recommendations
        for resource_type, data in utilization.items():
            if data['overall_utilization_rate'] < 60:
                recommendations.append({
                    'category': 'utilization_improvement',
                    'priority': 'medium',
                    'recommendation': f'Improve {resource_type} utilization through better scheduling and allocation',
                    'expected_benefit': f'Increase efficiency by {80 - data["overall_utilization_rate"]:.0f}%'
                })
            
            elif data['overall_utilization_rate'] > 90:
                recommendations.append({
                    'category': 'capacity_expansion',
                    'priority': 'high',
                    'recommendation': f'Consider expanding {resource_type} capacity to meet demand',
                    'expected_benefit': 'Reduce bottlenecks and improve service quality'
                })
        
        # Opportunity-based recommendations
        underutilized_count = len([o for o in opportunities if o['type'] == 'underutilization'])
        if underutilized_count > 0:
            recommendations.append({
                'category': 'resource_reallocation',
                'priority': 'medium',
                'recommendation': f'Reallocate {underutilized_count} underutilized resources to high-demand areas',
                'expected_benefit': 'Improve overall resource efficiency'
            })
        
        maintenance_needed = len([o for o in opportunities if o['type'] == 'maintenance_needed'])
        if maintenance_needed > 0:
            recommendations.append({
                'category': 'maintenance_priority',
                'priority': 'high',
                'recommendation': f'Prioritize maintenance for {maintenance_needed} resources in poor condition',
                'expected_benefit': 'Prevent service disruptions and extend asset life'
            })
        
        return recommendations

# Initialize engine
education_engine = EducationalIntelligenceEngine()

# Routes
@app.route('/education')
def education_dashboard():
    """Educational Institution Management dashboard"""
    
    recent_institutions = EducationalInstitution.query.order_by(EducationalInstitution.created_at.desc()).limit(10).all()
    
    return render_template('education/dashboard.html',
                         institutions=recent_institutions)

@app.route('/education/api/student-performance', methods=['POST'])
def predict_student_performance():
    """API endpoint for student performance prediction"""
    
    data = request.get_json()
    institution_id = data.get('institution_id')
    
    if not institution_id:
        return jsonify({'error': 'Institution ID required'}), 400
    
    predictions = education_engine.predict_student_performance(institution_id)
    return jsonify(predictions)

@app.route('/education/api/resource-optimization', methods=['POST'])
def optimize_resources():
    """API endpoint for resource optimization"""
    
    data = request.get_json()
    institution_id = data.get('institution_id')
    
    if not institution_id:
        return jsonify({'error': 'Institution ID required'}), 400
    
    optimization = education_engine.optimize_resource_allocation(institution_id)
    return jsonify(optimization)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if EducationalInstitution.query.count() == 0:
        sample_institution = EducationalInstitution(
            institution_id='EDU_DEMO_001',
            institution_name='Demo High School',
            institution_type=InstitutionType.HIGH_SCHOOL,
            location='Academic District',
            student_capacity=1200,
            current_enrollment=1050,
            faculty_count=75,
            graduation_rate=92.5
        )
        
        db.session.add(sample_institution)
        db.session.commit()
        logger.info("Sample educational institution data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5025, debug=True)