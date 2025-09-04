"""
Professional Services Automation Hub - Complete AI Platform for Service Businesses
$15B+ Value Potential - Project Management, Billing, Client Relations & Knowledge Management
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
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "professional-services-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///professional_services.db")

db.init_app(app)

# Professional Services Enums
class ServiceType(Enum):
    CONSULTING = "consulting"
    LEGAL = "legal"
    ACCOUNTING = "accounting"
    MARKETING = "marketing"
    DESIGN = "design"
    TECHNOLOGY = "technology"
    ENGINEERING = "engineering"

class ProjectStatus(Enum):
    PROPOSAL = "proposal"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class BillingModel(Enum):
    HOURLY = "hourly"
    FIXED_PRICE = "fixed_price"
    RETAINER = "retainer"
    VALUE_BASED = "value_based"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Data Models
class ProfessionalServicesFirm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firm_id = db.Column(db.String(100), unique=True, nullable=False)
    firm_name = db.Column(db.String(200), nullable=False)
    service_type = db.Column(db.Enum(ServiceType), nullable=False)
    
    # Firm Details
    employee_count = db.Column(db.Integer, default=1)
    specializations = db.Column(db.JSON)  # List of specialized services
    target_industries = db.Column(db.JSON)
    
    # Performance Metrics
    monthly_revenue = db.Column(db.Float, default=0.0)
    active_clients = db.Column(db.Integer, default=0)
    average_project_value = db.Column(db.Float, default=0.0)
    utilization_rate = db.Column(db.Float, default=75.0)  # percentage
    
    # Business Configuration
    standard_hourly_rate = db.Column(db.Float, default=150.0)
    preferred_billing_model = db.Column(db.Enum(BillingModel), default=BillingModel.HOURLY)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(100), unique=True, nullable=False)
    firm_id = db.Column(db.String(100), db.ForeignKey('professional_services_firm.firm_id'), nullable=False)
    
    # Client Details
    company_name = db.Column(db.String(200), nullable=False)
    industry = db.Column(db.String(100))
    company_size = db.Column(db.String(50))  # startup, small, medium, large, enterprise
    primary_contact_name = db.Column(db.String(200))
    primary_contact_email = db.Column(db.String(200))
    
    # Relationship Management
    relationship_stage = db.Column(db.String(50))  # prospect, new, established, strategic
    satisfaction_score = db.Column(db.Float, default=8.0)  # 1-10 scale
    communication_frequency = db.Column(db.String(50))  # weekly, monthly, quarterly
    
    # Financial Information
    total_revenue = db.Column(db.Float, default=0.0)
    average_project_size = db.Column(db.Float, default=0.0)
    payment_terms = db.Column(db.String(50), default='NET30')
    credit_rating = db.Column(db.String(10))
    
    # AI Insights
    churn_probability = db.Column(db.Float, default=0.0)
    upsell_potential = db.Column(db.Float, default=50.0)
    recommended_services = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.String(100), unique=True, nullable=False)
    firm_id = db.Column(db.String(100), db.ForeignKey('professional_services_firm.firm_id'), nullable=False)
    client_id = db.Column(db.String(100), db.ForeignKey('client.client_id'), nullable=False)
    
    # Project Details
    project_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    project_type = db.Column(db.String(100))
    status = db.Column(db.Enum(ProjectStatus), default=ProjectStatus.PROPOSAL)
    
    # Timeline
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    estimated_hours = db.Column(db.Float, default=0.0)
    actual_hours = db.Column(db.Float, default=0.0)
    
    # Financial
    billing_model = db.Column(db.Enum(BillingModel), default=BillingModel.HOURLY)
    estimated_value = db.Column(db.Float, default=0.0)
    actual_revenue = db.Column(db.Float, default=0.0)
    hourly_rate = db.Column(db.Float, default=150.0)
    
    # Progress Tracking
    completion_percentage = db.Column(db.Float, default=0.0)
    milestones = db.Column(db.JSON)  # List of milestone objects
    deliverables = db.Column(db.JSON)  # List of deliverable objects
    
    # AI Analysis
    risk_score = db.Column(db.Float, default=25.0)  # 0-100
    success_probability = db.Column(db.Float, default=85.0)
    predicted_completion_date = db.Column(db.Date)
    resource_optimization_suggestions = db.Column(db.JSON)
    
    # Team Assignment
    assigned_team_members = db.Column(db.JSON)  # List of employee IDs
    project_manager_id = db.Column(db.String(100))
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class TimeEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.String(100), unique=True, nullable=False)
    project_id = db.Column(db.String(100), db.ForeignKey('project.project_id'), nullable=False)
    employee_id = db.Column(db.String(100), nullable=False)
    
    # Time Details
    date = db.Column(db.Date, nullable=False)
    hours_worked = db.Column(db.Float, nullable=False)
    task_description = db.Column(db.Text)
    task_category = db.Column(db.String(100))  # research, design, development, meeting, etc.
    
    # Billing Information
    billable = db.Column(db.Boolean, default=True)
    hourly_rate = db.Column(db.Float, default=150.0)
    amount = db.Column(db.Float, default=0.0)
    
    # Quality Metrics
    efficiency_score = db.Column(db.Float, default=80.0)  # 0-100
    quality_rating = db.Column(db.Float, default=85.0)  # 0-100
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(100), unique=True, nullable=False)
    firm_id = db.Column(db.String(100), db.ForeignKey('professional_services_firm.firm_id'), nullable=False)
    
    # Employee Details
    name = db.Column(db.String(200), nullable=False)
    position = db.Column(db.String(100))
    department = db.Column(db.String(100))
    seniority_level = db.Column(db.String(50))  # junior, mid, senior, principal
    
    # Skills and Expertise
    skills = db.Column(db.JSON)  # List of skills
    certifications = db.Column(db.JSON)
    specializations = db.Column(db.JSON)
    
    # Performance Metrics
    hourly_rate = db.Column(db.Float, default=150.0)
    utilization_rate = db.Column(db.Float, default=75.0)  # percentage
    average_quality_score = db.Column(db.Float, default=85.0)
    client_satisfaction_score = db.Column(db.Float, default=8.5)
    
    # Capacity Management
    weekly_capacity_hours = db.Column(db.Float, default=40.0)
    current_workload_hours = db.Column(db.Float, default=30.0)
    availability_status = db.Column(db.String(50), default='available')  # available, busy, overloaded
    
    # AI Insights
    performance_trend = db.Column(db.String(50))  # improving, stable, declining
    skill_development_recommendations = db.Column(db.JSON)
    optimal_project_types = db.Column(db.JSON)
    
    hire_date = db.Column(db.Date)

class KnowledgeBase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    knowledge_id = db.Column(db.String(100), unique=True, nullable=False)
    firm_id = db.Column(db.String(100), db.ForeignKey('professional_services_firm.firm_id'), nullable=False)
    
    # Knowledge Content
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100))
    tags = db.Column(db.JSON)  # List of tags
    
    # Metadata
    author_id = db.Column(db.String(100))
    project_id = db.Column(db.String(100))  # If derived from a project
    client_id = db.Column(db.String(100))  # If client-specific
    
    # Usage Analytics
    view_count = db.Column(db.Integer, default=0)
    usefulness_rating = db.Column(db.Float, default=7.5)  # 1-10 scale
    last_accessed = db.Column(db.DateTime)
    
    # AI Enhancement
    related_knowledge_ids = db.Column(db.JSON)  # Related documents
    ai_summary = db.Column(db.Text)
    key_insights = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# Professional Services Automation Engine
class ProfessionalServicesEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def optimize_project_management(self, firm_id: str) -> Dict[str, Any]:
        """Comprehensive project management optimization"""
        
        active_projects = Project.query.filter_by(firm_id=firm_id, status=ProjectStatus.ACTIVE).all()
        
        if not active_projects:
            return {'error': 'No active projects found'}
        
        optimization_results = []
        total_risk_reduction = 0
        
        for project in active_projects:
            project_analysis = self._analyze_project_optimization(project)
            optimization_results.append(project_analysis)
            total_risk_reduction += project_analysis.get('risk_reduction_potential', 0)
        
        # Resource allocation optimization
        resource_optimization = self._optimize_resource_allocation(firm_id, active_projects)
        
        # Timeline optimization
        timeline_optimization = self._optimize_project_timelines(active_projects)
        
        return {
            'firm_id': firm_id,
            'projects_analyzed': len(active_projects),
            'total_risk_reduction_potential': total_risk_reduction,
            'project_optimizations': optimization_results,
            'resource_optimization': resource_optimization,
            'timeline_optimization': timeline_optimization,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_project_optimization(self, project: Project) -> Dict[str, Any]:
        """Analyze optimization opportunities for individual project"""
        
        # Calculate project health metrics
        health_metrics = self._calculate_project_health(project)
        
        # Identify bottlenecks
        bottlenecks = self._identify_project_bottlenecks(project)
        
        # Resource utilization analysis
        resource_analysis = self._analyze_resource_utilization(project)
        
        # Generate optimization recommendations
        optimizations = self._generate_project_optimizations(project, health_metrics, bottlenecks)
        
        return {
            'project_id': project.project_id,
            'project_name': project.project_name,
            'current_health_score': health_metrics['overall_score'],
            'completion_status': project.completion_percentage,
            'bottlenecks': bottlenecks,
            'resource_utilization': resource_analysis,
            'optimization_recommendations': optimizations,
            'risk_reduction_potential': health_metrics['risk_score'] - max(10, health_metrics['risk_score'] * 0.7)
        }
    
    def _calculate_project_health(self, project: Project) -> Dict[str, Any]:
        """Calculate comprehensive project health metrics"""
        
        # Timeline health
        if project.start_date and project.end_date:
            total_days = (project.end_date - project.start_date).days
            elapsed_days = (datetime.now().date() - project.start_date).days
            timeline_progress = (elapsed_days / total_days) * 100 if total_days > 0 else 0
            timeline_vs_completion = timeline_progress - project.completion_percentage
        else:
            timeline_vs_completion = 0
        
        # Budget health
        budget_variance = 0
        if project.estimated_value > 0:
            budget_variance = ((project.actual_revenue - project.estimated_value) / project.estimated_value) * 100
        
        # Resource health
        hour_variance = 0
        if project.estimated_hours > 0:
            hour_variance = ((project.actual_hours - project.estimated_hours) / project.estimated_hours) * 100
        
        # Calculate overall health score
        timeline_score = max(0, 100 - abs(timeline_vs_completion))
        budget_score = max(0, 100 - abs(budget_variance))
        resource_score = max(0, 100 - abs(hour_variance))
        
        overall_score = (timeline_score + budget_score + resource_score) / 3
        
        return {
            'overall_score': overall_score,
            'timeline_score': timeline_score,
            'budget_score': budget_score,
            'resource_score': resource_score,
            'timeline_variance': timeline_vs_completion,
            'budget_variance': budget_variance,
            'hour_variance': hour_variance,
            'risk_score': project.risk_score
        }
    
    def _identify_project_bottlenecks(self, project: Project) -> List[Dict[str, Any]]:
        """Identify project bottlenecks and constraints"""
        
        bottlenecks = []
        
        # Check timeline bottlenecks
        if project.completion_percentage < 50 and project.end_date:
            days_remaining = (project.end_date - datetime.now().date()).days
            if days_remaining < 30:
                bottlenecks.append({
                    'type': 'timeline',
                    'severity': 'high',
                    'description': 'Project behind schedule with limited time remaining',
                    'impact': 'delivery_risk'
                })
        
        # Check resource bottlenecks
        if project.actual_hours > project.estimated_hours * 1.2:
            bottlenecks.append({
                'type': 'resource',
                'severity': 'medium',
                'description': 'Resource consumption exceeding estimates',
                'impact': 'budget_overrun'
            })
        
        # Check milestone bottlenecks
        milestones = project.milestones or []
        overdue_milestones = [m for m in milestones if m.get('due_date') and 
                             datetime.strptime(m['due_date'], '%Y-%m-%d').date() < datetime.now().date() and 
                             not m.get('completed')]
        
        if overdue_milestones:
            bottlenecks.append({
                'type': 'milestone',
                'severity': 'high',
                'description': f'{len(overdue_milestones)} milestones overdue',
                'impact': 'client_satisfaction'
            })
        
        return bottlenecks
    
    def _analyze_resource_utilization(self, project: Project) -> Dict[str, Any]:
        """Analyze resource utilization for project"""
        
        # Get time entries for project
        time_entries = TimeEntry.query.filter_by(project_id=project.project_id).all()
        
        if not time_entries:
            return {'error': 'No time entries found'}
        
        # Calculate utilization metrics
        total_hours = sum(entry.hours_worked for entry in time_entries)
        billable_hours = sum(entry.hours_worked for entry in time_entries if entry.billable)
        
        # Team member utilization
        team_utilization = {}
        for entry in time_entries:
            employee_id = entry.employee_id
            if employee_id not in team_utilization:
                team_utilization[employee_id] = {'total_hours': 0, 'billable_hours': 0}
            
            team_utilization[employee_id]['total_hours'] += entry.hours_worked
            team_utilization[employee_id]['billable_hours'] += entry.hours_worked if entry.billable else 0
        
        # Calculate efficiency metrics
        billable_percentage = (billable_hours / total_hours * 100) if total_hours > 0 else 0
        average_efficiency = np.mean([entry.efficiency_score for entry in time_entries])
        
        return {
            'total_hours_logged': total_hours,
            'billable_hours': billable_hours,
            'billable_percentage': billable_percentage,
            'team_member_count': len(team_utilization),
            'average_efficiency_score': average_efficiency,
            'utilization_by_member': team_utilization
        }
    
    def _generate_project_optimizations(self, project: Project, health_metrics: Dict, bottlenecks: List) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Timeline optimizations
        if health_metrics['timeline_variance'] > 10:
            recommendations.append("Consider reallocating resources to critical path tasks")
            recommendations.append("Review and potentially adjust project timeline with client")
        
        # Budget optimizations
        if health_metrics['budget_variance'] < -15:
            recommendations.append("Implement stricter scope management to control costs")
            recommendations.append("Review hourly rates and billing efficiency")
        
        # Resource optimizations
        if health_metrics['hour_variance'] > 20:
            recommendations.append("Conduct task complexity review and adjust estimates")
            recommendations.append("Provide additional training or support to team members")
        
        # Bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'timeline':
                recommendations.append("Implement daily standups to accelerate progress")
            elif bottleneck['type'] == 'resource':
                recommendations.append("Consider bringing in additional resources")
            elif bottleneck['type'] == 'milestone':
                recommendations.append("Prioritize overdue milestones and reassign if necessary")
        
        return recommendations
    
    def _optimize_resource_allocation(self, firm_id: str, projects: List[Project]) -> Dict[str, Any]:
        """Optimize resource allocation across projects"""
        
        # Get all employees
        employees = Employee.query.filter_by(firm_id=firm_id).all()
        
        # Calculate current workload distribution
        workload_analysis = {}
        for employee in employees:
            workload_analysis[employee.employee_id] = {
                'name': employee.name,
                'current_hours': employee.current_workload_hours,
                'capacity_hours': employee.weekly_capacity_hours,
                'utilization_rate': employee.utilization_rate,
                'skills': employee.skills or [],
                'projects': []
            }
        
        # Analyze project assignments
        for project in projects:
            assigned_members = project.assigned_team_members or []
            for member_id in assigned_members:
                if member_id in workload_analysis:
                    workload_analysis[member_id]['projects'].append({
                        'project_id': project.project_id,
                        'project_name': project.project_name,
                        'completion': project.completion_percentage
                    })
        
        # Identify optimization opportunities
        overloaded_employees = [emp_id for emp_id, data in workload_analysis.items() 
                               if data['utilization_rate'] > 90]
        underutilized_employees = [emp_id for emp_id, data in workload_analysis.items() 
                                  if data['utilization_rate'] < 60]
        
        return {
            'total_employees': len(employees),
            'overloaded_employees': len(overloaded_employees),
            'underutilized_employees': len(underutilized_employees),
            'average_utilization': np.mean([emp.utilization_rate for emp in employees]),
            'workload_distribution': workload_analysis,
            'rebalancing_recommendations': self._generate_rebalancing_recommendations(
                overloaded_employees, underutilized_employees, workload_analysis
            )
        }
    
    def _generate_rebalancing_recommendations(self, overloaded: List[str], underutilized: List[str], 
                                            workload_data: Dict) -> List[str]:
        """Generate workload rebalancing recommendations"""
        
        recommendations = []
        
        if overloaded:
            recommendations.append(f"Redistribute workload from {len(overloaded)} overloaded team members")
        
        if underutilized:
            recommendations.append(f"Increase utilization for {len(underutilized)} underutilized team members")
        
        # Skill-based matching recommendations
        for overloaded_id in overloaded[:3]:  # Top 3 overloaded
            overloaded_skills = workload_data[overloaded_id]['skills']
            
            for underutilized_id in underutilized:
                underutilized_skills = workload_data[underutilized_id]['skills']
                skill_overlap = set(overloaded_skills) & set(underutilized_skills)
                
                if skill_overlap:
                    recommendations.append(
                        f"Transfer tasks requiring {list(skill_overlap)[0]} skills from "
                        f"{workload_data[overloaded_id]['name']} to {workload_data[underutilized_id]['name']}"
                    )
        
        return recommendations
    
    def _optimize_project_timelines(self, projects: List[Project]) -> Dict[str, Any]:
        """Optimize project timelines and dependencies"""
        
        timeline_issues = []
        optimization_opportunities = []
        
        for project in projects:
            # Check for timeline optimization opportunities
            if project.end_date and project.completion_percentage > 0:
                projected_completion = self._calculate_projected_completion(project)
                
                if projected_completion > project.end_date:
                    timeline_issues.append({
                        'project_id': project.project_id,
                        'project_name': project.project_name,
                        'original_end_date': project.end_date.isoformat(),
                        'projected_end_date': projected_completion.isoformat(),
                        'delay_days': (projected_completion - project.end_date).days
                    })
                
                # Identify acceleration opportunities
                if project.completion_percentage < 30 and project.risk_score < 30:
                    optimization_opportunities.append({
                        'project_id': project.project_id,
                        'project_name': project.project_name,
                        'opportunity': 'early_completion',
                        'potential_savings': 'Resource reallocation possible'
                    })
        
        return {
            'projects_at_risk': len(timeline_issues),
            'acceleration_opportunities': len(optimization_opportunities),
            'timeline_issues': timeline_issues,
            'optimization_opportunities': optimization_opportunities
        }
    
    def _calculate_projected_completion(self, project: Project) -> datetime:
        """Calculate projected completion date based on current progress"""
        
        if project.completion_percentage > 0 and project.start_date:
            days_elapsed = (datetime.now().date() - project.start_date).days
            total_days_estimated = days_elapsed / (project.completion_percentage / 100)
            projected_end = project.start_date + timedelta(days=total_days_estimated)
            return projected_end
        
        return project.end_date or datetime.now().date()
    
    def generate_intelligent_billing(self, firm_id: str) -> Dict[str, Any]:
        """AI-powered billing optimization and automation"""
        
        # Get recent time entries
        recent_entries = TimeEntry.query.join(Project).filter(
            Project.firm_id == firm_id,
            TimeEntry.date >= datetime.now().date() - timedelta(days=30)
        ).all()
        
        if not recent_entries:
            return {'error': 'No recent time entries found'}
        
        # Analyze billing efficiency
        billing_analysis = self._analyze_billing_efficiency(recent_entries)
        
        # Identify billing optimization opportunities
        optimization_opportunities = self._identify_billing_optimizations(recent_entries)
        
        # Generate automated billing recommendations
        automation_recommendations = self._generate_billing_automation_recommendations(billing_analysis)
        
        return {
            'firm_id': firm_id,
            'analysis_period': '30 days',
            'total_billable_hours': sum(entry.hours_worked for entry in recent_entries if entry.billable),
            'total_revenue': sum(entry.amount for entry in recent_entries if entry.billable),
            'billing_efficiency_analysis': billing_analysis,
            'optimization_opportunities': optimization_opportunities,
            'automation_recommendations': automation_recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_billing_efficiency(self, time_entries: List[TimeEntry]) -> Dict[str, Any]:
        """Analyze billing efficiency and patterns"""
        
        total_hours = sum(entry.hours_worked for entry in time_entries)
        billable_hours = sum(entry.hours_worked for entry in time_entries if entry.billable)
        
        # Calculate efficiency metrics
        billable_ratio = (billable_hours / total_hours * 100) if total_hours > 0 else 0
        average_hourly_rate = np.mean([entry.hourly_rate for entry in time_entries if entry.billable])
        
        # Analyze by category
        category_analysis = {}
        for entry in time_entries:
            category = entry.task_category or 'other'
            if category not in category_analysis:
                category_analysis[category] = {'hours': 0, 'billable_hours': 0, 'revenue': 0}
            
            category_analysis[category]['hours'] += entry.hours_worked
            if entry.billable:
                category_analysis[category]['billable_hours'] += entry.hours_worked
                category_analysis[category]['revenue'] += entry.amount
        
        # Calculate realization rates
        realization_rate = 85.0  # Simplified - would calculate based on actual vs billed amounts
        
        return {
            'billable_ratio_percent': billable_ratio,
            'average_hourly_rate': average_hourly_rate,
            'realization_rate_percent': realization_rate,
            'category_breakdown': category_analysis,
            'efficiency_score': min(100, (billable_ratio + realization_rate) / 2)
        }
    
    def _identify_billing_optimizations(self, time_entries: List[TimeEntry]) -> List[Dict[str, Any]]:
        """Identify billing optimization opportunities"""
        
        opportunities = []
        
        # Low billable ratio identification
        non_billable_entries = [entry for entry in time_entries if not entry.billable]
        if len(non_billable_entries) > len(time_entries) * 0.3:
            opportunities.append({
                'type': 'billable_ratio',
                'description': 'High percentage of non-billable time',
                'impact': 'revenue_loss',
                'recommendation': 'Review non-billable activities and consider value-based billing'
            })
        
        # Rate optimization opportunities
        rate_variance = np.std([entry.hourly_rate for entry in time_entries if entry.billable])
        if rate_variance > 50:
            opportunities.append({
                'type': 'rate_standardization',
                'description': 'High variance in hourly rates',
                'impact': 'pricing_consistency',
                'recommendation': 'Standardize rates by role and experience level'
            })
        
        # Time tracking gaps
        entries_by_date = {}
        for entry in time_entries:
            date_str = entry.date.isoformat()
            entries_by_date[date_str] = entries_by_date.get(date_str, 0) + 1
        
        missing_days = len([date for date, count in entries_by_date.items() if count == 0])
        if missing_days > 5:
            opportunities.append({
                'type': 'time_tracking',
                'description': f'{missing_days} days with no time entries',
                'impact': 'revenue_leakage',
                'recommendation': 'Implement automated time tracking reminders'
            })
        
        return opportunities
    
    def _generate_billing_automation_recommendations(self, billing_analysis: Dict[str, Any]) -> List[str]:
        """Generate billing automation recommendations"""
        
        recommendations = []
        
        if billing_analysis['billable_ratio_percent'] < 70:
            recommendations.append("Implement automated billable time alerts for team members")
        
        if billing_analysis['efficiency_score'] < 80:
            recommendations.append("Set up automated invoice generation based on time entries")
        
        recommendations.extend([
            "Enable automatic time tracking for common tasks",
            "Implement smart billing rate suggestions based on project complexity",
            "Set up automated expense tracking and reimbursement",
            "Create automated billing reminders for overdue invoices",
            "Implement AI-powered time entry validation and correction"
        ])
        
        return recommendations

# Initialize engine
services_engine = ProfessionalServicesEngine()

# Routes
@app.route('/professional-services')
def services_dashboard():
    """Professional Services Automation dashboard"""
    
    recent_firms = ProfessionalServicesFirm.query.order_by(ProfessionalServicesFirm.created_at.desc()).limit(10).all()
    
    return render_template('professional_services/dashboard.html',
                         firms=recent_firms)

@app.route('/professional-services/api/project-optimization', methods=['POST'])
def optimize_projects():
    """API endpoint for project management optimization"""
    
    data = request.get_json()
    firm_id = data.get('firm_id')
    
    if not firm_id:
        return jsonify({'error': 'Firm ID required'}), 400
    
    optimization = services_engine.optimize_project_management(firm_id)
    return jsonify(optimization)

@app.route('/professional-services/api/billing-optimization', methods=['POST'])
def optimize_billing():
    """API endpoint for billing optimization"""
    
    data = request.get_json()
    firm_id = data.get('firm_id')
    
    if not firm_id:
        return jsonify({'error': 'Firm ID required'}), 400
    
    billing_optimization = services_engine.generate_intelligent_billing(firm_id)
    return jsonify(billing_optimization)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if ProfessionalServicesFirm.query.count() == 0:
        sample_firm = ProfessionalServicesFirm(
            firm_id='PROF_DEMO_001',
            firm_name='Demo Consulting Group',
            service_type=ServiceType.CONSULTING,
            employee_count=25,
            specializations=['digital_transformation', 'process_optimization'],
            monthly_revenue=450000,
            active_clients=15
        )
        
        db.session.add(sample_firm)
        db.session.commit()
        logger.info("Sample professional services data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5022, debug=True)