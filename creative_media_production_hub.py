"""
Creative & Media Production Hub - Complete AI-Powered Creative Services Platform
$22B+ Value Potential - Content Creation, Project Management & Brand Consistency
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
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "creative-media-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///creative_media.db")

db.init_app(app)

# Creative Media Enums
class AgencyType(Enum):
    MARKETING = "marketing"
    ADVERTISING = "advertising"
    DESIGN = "design"
    VIDEO_PRODUCTION = "video_production"
    DIGITAL_CONTENT = "digital_content"
    PHOTOGRAPHY = "photography"

class ProjectStatus(Enum):
    PROPOSAL = "proposal"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    REVISION = "revision"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ContentType(Enum):
    VIDEO = "video"
    GRAPHIC_DESIGN = "graphic_design"
    PHOTOGRAPHY = "photography"
    COPYWRITING = "copywriting"
    WEB_DESIGN = "web_design"
    SOCIAL_MEDIA = "social_media"
    ANIMATION = "animation"

# Data Models
class CreativeAgency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agency_id = db.Column(db.String(100), unique=True, nullable=False)
    agency_name = db.Column(db.String(200), nullable=False)
    agency_type = db.Column(db.Enum(AgencyType), nullable=False)
    
    # Agency Details
    location = db.Column(db.String(200))
    team_size = db.Column(db.Integer, default=10)
    specializations = db.Column(db.JSON)  # List of creative specializations
    
    # Performance Metrics
    active_projects = db.Column(db.Integer, default=0)
    monthly_revenue = db.Column(db.Float, default=0.0)
    client_retention_rate = db.Column(db.Float, default=85.0)  # percentage
    average_project_value = db.Column(db.Float, default=15000.0)
    
    # Quality Metrics
    client_satisfaction_score = db.Column(db.Float, default=4.2)  # 1-5 scale
    project_on_time_rate = db.Column(db.Float, default=88.0)  # percentage
    revision_rate = db.Column(db.Float, default=25.0)  # percentage requiring revisions
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(100), unique=True, nullable=False)
    agency_id = db.Column(db.String(100), db.ForeignKey('creative_agency.agency_id'), nullable=False)
    
    # Client Information
    company_name = db.Column(db.String(200), nullable=False)
    industry = db.Column(db.String(100))
    company_size = db.Column(db.String(50))  # startup, small, medium, large, enterprise
    primary_contact = db.Column(db.String(200))
    email = db.Column(db.String(200))
    
    # Brand Information
    brand_guidelines = db.Column(db.JSON)  # Brand colors, fonts, style guide
    target_audience = db.Column(db.JSON)
    brand_voice = db.Column(db.Text)
    
    # Relationship Metrics
    total_projects = db.Column(db.Integer, default=0)
    total_revenue = db.Column(db.Float, default=0.0)
    satisfaction_score = db.Column(db.Float, default=4.0)  # 1-5 scale
    payment_reliability = db.Column(db.Float, default=95.0)  # percentage
    
    # Preferences
    preferred_content_types = db.Column(db.JSON)
    communication_frequency = db.Column(db.String(50), default='weekly')
    approval_process = db.Column(db.JSON)  # Steps in client approval process
    
    # AI Insights
    lifetime_value = db.Column(db.Float, default=0.0)
    churn_probability = db.Column(db.Float, default=0.15)  # 0-1
    upsell_opportunities = db.Column(db.JSON)
    
    start_date = db.Column(db.Date)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.String(100), unique=True, nullable=False)
    agency_id = db.Column(db.String(100), db.ForeignKey('creative_agency.agency_id'), nullable=False)
    client_id = db.Column(db.String(100), db.ForeignKey('client.client_id'), nullable=False)
    
    # Project Details
    project_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    content_type = db.Column(db.Enum(ContentType), nullable=False)
    status = db.Column(db.Enum(ProjectStatus), default=ProjectStatus.PROPOSAL)
    
    # Timeline
    start_date = db.Column(db.Date)
    deadline = db.Column(db.Date)
    estimated_hours = db.Column(db.Float, default=40.0)
    actual_hours = db.Column(db.Float, default=0.0)
    
    # Financial
    project_value = db.Column(db.Float, nullable=False)
    costs_incurred = db.Column(db.Float, default=0.0)
    profit_margin = db.Column(db.Float, default=40.0)  # percentage
    
    # Creative Requirements
    creative_brief = db.Column(db.JSON)  # Detailed creative requirements
    deliverables = db.Column(db.JSON)  # List of expected deliverables
    style_references = db.Column(db.JSON)  # Reference materials and inspiration
    
    # Team Assignment
    creative_director_id = db.Column(db.String(100))
    assigned_team = db.Column(db.JSON)  # List of team member IDs and roles
    
    # Progress Tracking
    completion_percentage = db.Column(db.Float, default=0.0)
    milestones = db.Column(db.JSON)  # Project milestones and deadlines
    revision_count = db.Column(db.Integer, default=0)
    
    # Quality Control
    brand_compliance_score = db.Column(db.Float, default=85.0)  # 0-100
    client_feedback = db.Column(db.JSON)
    internal_quality_score = db.Column(db.Float, default=80.0)  # 0-100
    
    # AI Analysis
    risk_score = db.Column(db.Float, default=25.0)  # 0-100
    success_probability = db.Column(db.Float, default=85.0)  # percentage
    optimization_suggestions = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CreativeAsset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset_id = db.Column(db.String(100), unique=True, nullable=False)
    project_id = db.Column(db.String(100), db.ForeignKey('project.project_id'), nullable=False)
    
    # Asset Information
    asset_name = db.Column(db.String(200), nullable=False)
    asset_type = db.Column(db.String(100))  # logo, banner, video, photo, etc.
    file_format = db.Column(db.String(20))  # jpg, png, mp4, ai, psd, etc.
    file_size_mb = db.Column(db.Float, default=0.0)
    
    # Versions and Approval
    version_number = db.Column(db.Float, default=1.0)
    approval_status = db.Column(db.String(50), default='pending')  # pending, approved, rejected
    approval_date = db.Column(db.DateTime)
    
    # Creative Details
    dimensions = db.Column(db.String(50))  # 1920x1080, 8.5x11, etc.
    color_palette = db.Column(db.JSON)  # Primary colors used
    fonts_used = db.Column(db.JSON)
    creation_software = db.Column(db.String(100))
    
    # Usage Rights
    usage_rights = db.Column(db.JSON)  # Where and how asset can be used
    expiration_date = db.Column(db.Date)
    copyright_info = db.Column(db.Text)
    
    # Performance Tracking
    usage_count = db.Column(db.Integer, default=0)
    performance_metrics = db.Column(db.JSON)  # CTR, engagement, etc.
    
    # AI Analysis
    brand_consistency_score = db.Column(db.Float, default=85.0)  # 0-100
    visual_appeal_score = db.Column(db.Float, default=80.0)  # 0-100
    optimization_recommendations = db.Column(db.JSON)
    
    created_by = db.Column(db.String(100))  # Creator ID
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TeamMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    member_id = db.Column(db.String(100), unique=True, nullable=False)
    agency_id = db.Column(db.String(100), db.ForeignKey('creative_agency.agency_id'), nullable=False)
    
    # Personal Information
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    role = db.Column(db.String(100))  # designer, copywriter, art_director, etc.
    seniority_level = db.Column(db.String(50))  # junior, mid, senior, lead
    
    # Skills and Specializations
    skills = db.Column(db.JSON)  # List of skills and proficiency levels
    software_proficiency = db.Column(db.JSON)  # Adobe Creative Suite, etc.
    specializations = db.Column(db.JSON)  # Brand design, motion graphics, etc.
    
    # Performance Metrics
    hourly_rate = db.Column(db.Float, default=75.0)
    productivity_score = db.Column(db.Float, default=85.0)  # 0-100
    quality_score = db.Column(db.Float, default=88.0)  # 0-100
    client_satisfaction = db.Column(db.Float, default=4.3)  # 1-5 scale
    
    # Workload Management
    current_projects = db.Column(db.JSON)  # Current project assignments
    capacity_percentage = db.Column(db.Float, default=80.0)  # Current workload
    availability_calendar = db.Column(db.JSON)
    
    # Growth and Development
    training_completed = db.Column(db.JSON)
    certifications = db.Column(db.JSON)
    career_goals = db.Column(db.Text)
    
    # AI Insights
    performance_trend = db.Column(db.String(50), default='stable')  # improving, stable, declining
    skill_development_recommendations = db.Column(db.JSON)
    optimal_project_types = db.Column(db.JSON)
    
    hire_date = db.Column(db.Date)

class ContentLibrary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.String(100), unique=True, nullable=False)
    agency_id = db.Column(db.String(100), db.ForeignKey('creative_agency.agency_id'), nullable=False)
    
    # Content Information
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.Enum(ContentType), nullable=False)
    category = db.Column(db.String(100))  # template, stock_photo, brand_element, etc.
    
    # Content Details
    description = db.Column(db.Text)
    tags = db.Column(db.JSON)  # Searchable tags
    file_path = db.Column(db.String(500))
    thumbnail_path = db.Column(db.String(500))
    
    # Usage and Performance
    download_count = db.Column(db.Integer, default=0)
    usage_in_projects = db.Column(db.JSON)  # Projects that used this content
    popularity_score = db.Column(db.Float, default=50.0)  # 0-100
    
    # Metadata
    created_by = db.Column(db.String(100))
    creation_date = db.Column(db.Date)
    last_modified = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Rights and Licensing
    license_type = db.Column(db.String(100))  # royalty_free, licensed, custom
    usage_restrictions = db.Column(db.Text)
    
    # AI Enhancement
    auto_generated_tags = db.Column(db.JSON)
    style_similarity_score = db.Column(db.Float, default=0.0)
    recommended_usage = db.Column(db.JSON)

# Creative Intelligence Engine
class CreativeIntelligenceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def analyze_project_optimization(self, agency_id: str) -> Dict[str, Any]:
        """AI-powered project management and timeline optimization"""
        
        # Get active projects
        active_projects = Project.query.filter_by(agency_id=agency_id)\
                                     .filter(Project.status.in_([ProjectStatus.IN_PROGRESS, ProjectStatus.REVIEW]))\
                                     .all()
        
        if not active_projects:
            return {'error': 'No active projects found'}
        
        optimization_results = []
        total_risk_reduction = 0
        
        for project in active_projects:
            project_analysis = self._analyze_individual_project(project)
            optimization_results.append(project_analysis)
            total_risk_reduction += project_analysis.get('risk_reduction_potential', 0)
        
        # Resource allocation analysis
        resource_analysis = self._analyze_resource_allocation(agency_id, active_projects)
        
        # Timeline optimization
        timeline_optimization = self._optimize_project_timelines(active_projects)
        
        return {
            'agency_id': agency_id,
            'active_projects_count': len(active_projects),
            'total_risk_reduction_potential': total_risk_reduction,
            'project_analyses': optimization_results,
            'resource_allocation': resource_analysis,
            'timeline_optimization': timeline_optimization,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_individual_project(self, project: Project) -> Dict[str, Any]:
        """Analyze individual project for optimization opportunities"""
        
        # Calculate project health metrics
        timeline_health = self._calculate_timeline_health(project)
        budget_health = self._calculate_budget_health(project)
        quality_metrics = self._calculate_quality_metrics(project)
        
        # Risk assessment
        risk_factors = self._assess_project_risks(project, timeline_health, budget_health)
        
        # Generate recommendations
        recommendations = self._generate_project_recommendations(project, risk_factors)
        
        return {
            'project_id': project.project_id,
            'project_name': project.project_name,
            'content_type': project.content_type.value,
            'timeline_health': timeline_health,
            'budget_health': budget_health,
            'quality_metrics': quality_metrics,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'risk_reduction_potential': max(0, project.risk_score - 15)  # Potential to reduce risk to 15%
        }
    
    def _calculate_timeline_health(self, project: Project) -> Dict[str, Any]:
        """Calculate project timeline health metrics"""
        
        if not project.start_date or not project.deadline:
            return {'status': 'no_timeline_data'}
        
        total_days = (project.deadline - project.start_date).days
        elapsed_days = (datetime.now().date() - project.start_date).days
        remaining_days = (project.deadline - datetime.now().date()).days
        
        # Calculate progress vs time
        time_progress = (elapsed_days / total_days * 100) if total_days > 0 else 0
        work_progress = project.completion_percentage
        progress_variance = work_progress - time_progress
        
        # Determine timeline status
        if remaining_days < 0:
            status = 'overdue'
        elif progress_variance < -20:
            status = 'behind_schedule'
        elif progress_variance > 10:
            status = 'ahead_of_schedule'
        else:
            status = 'on_track'
        
        return {
            'status': status,
            'time_progress_percent': time_progress,
            'work_progress_percent': work_progress,
            'progress_variance': progress_variance,
            'remaining_days': remaining_days,
            'estimated_completion_date': self._estimate_completion_date(project)
        }
    
    def _estimate_completion_date(self, project: Project) -> str:
        """Estimate project completion date based on current progress"""
        
        if project.completion_percentage > 0:
            start_date = project.start_date
            elapsed_days = (datetime.now().date() - start_date).days
            total_days_estimated = elapsed_days / (project.completion_percentage / 100)
            estimated_completion = start_date + timedelta(days=total_days_estimated)
            return estimated_completion.isoformat()
        
        return project.deadline.isoformat() if project.deadline else 'unknown'
    
    def _calculate_budget_health(self, project: Project) -> Dict[str, Any]:
        """Calculate project budget health"""
        
        budget_utilization = (project.costs_incurred / project.project_value * 100) if project.project_value > 0 else 0
        remaining_budget = project.project_value - project.costs_incurred
        
        # Calculate budget efficiency
        if project.completion_percentage > 0:
            expected_costs = project.project_value * (project.completion_percentage / 100) * 0.7  # Assume 70% of budget for completion
            budget_variance = project.costs_incurred - expected_costs
        else:
            budget_variance = 0
        
        # Determine budget status
        if budget_utilization > 90:
            status = 'over_budget'
        elif budget_variance > project.project_value * 0.15:
            status = 'tracking_high'
        elif budget_variance < -project.project_value * 0.1:
            status = 'under_budget'
        else:
            status = 'on_track'
        
        return {
            'status': status,
            'budget_utilization_percent': budget_utilization,
            'remaining_budget': remaining_budget,
            'budget_variance': budget_variance,
            'projected_final_cost': project.costs_incurred / (project.completion_percentage / 100) if project.completion_percentage > 0 else project.project_value
        }
    
    def _calculate_quality_metrics(self, project: Project) -> Dict[str, Any]:
        """Calculate project quality metrics"""
        
        return {
            'brand_compliance_score': project.brand_compliance_score,
            'internal_quality_score': project.internal_quality_score,
            'revision_count': project.revision_count,
            'client_satisfaction_trend': 'positive' if project.revision_count < 3 else 'needs_attention'
        }
    
    def _assess_project_risks(self, project: Project, timeline_health: Dict, budget_health: Dict) -> List[Dict[str, Any]]:
        """Assess various project risk factors"""
        
        risks = []
        
        # Timeline risks
        if timeline_health['status'] == 'behind_schedule':
            risks.append({
                'type': 'timeline',
                'severity': 'high' if timeline_health['progress_variance'] < -30 else 'medium',
                'description': 'Project behind schedule',
                'impact': 'Client satisfaction and future projects at risk'
            })
        
        # Budget risks
        if budget_health['status'] in ['over_budget', 'tracking_high']:
            risks.append({
                'type': 'budget',
                'severity': 'high' if budget_health['budget_utilization_percent'] > 95 else 'medium',
                'description': 'Budget overrun risk',
                'impact': 'Reduced profit margin'
            })
        
        # Quality risks
        if project.revision_count > 3:
            risks.append({
                'type': 'quality',
                'severity': 'medium',
                'description': 'High revision count indicates quality issues',
                'impact': 'Timeline delays and client satisfaction issues'
            })
        
        # Resource risks
        if project.actual_hours > project.estimated_hours * 1.3:
            risks.append({
                'type': 'resource',
                'severity': 'medium',
                'description': 'Resource consumption exceeding estimates',
                'impact': 'Team burnout and capacity issues'
            })
        
        return risks
    
    def _generate_project_recommendations(self, project: Project, risk_factors: List[Dict]) -> List[str]:
        """Generate specific recommendations for project optimization"""
        
        recommendations = []
        
        # Timeline recommendations
        timeline_risks = [r for r in risk_factors if r['type'] == 'timeline']
        if timeline_risks:
            recommendations.append("Reallocate resources to critical path tasks")
            recommendations.append("Consider scope adjustment discussions with client")
        
        # Budget recommendations
        budget_risks = [r for r in risk_factors if r['type'] == 'budget']
        if budget_risks:
            recommendations.append("Implement stricter cost control measures")
            recommendations.append("Review resource allocation efficiency")
        
        # Quality recommendations
        quality_risks = [r for r in risk_factors if r['type'] == 'quality']
        if quality_risks:
            recommendations.append("Enhance quality review process")
            recommendations.append("Increase client communication frequency")
        
        # General recommendations
        if project.completion_percentage < 30:
            recommendations.append("Establish clear milestone checkpoints")
        
        return recommendations
    
    def _analyze_resource_allocation(self, agency_id: str, projects: List[Project]) -> Dict[str, Any]:
        """Analyze team resource allocation across projects"""
        
        team_members = TeamMember.query.filter_by(agency_id=agency_id).all()
        
        # Calculate workload distribution
        workload_analysis = {}
        for member in team_members:
            workload_analysis[member.member_id] = {
                'name': member.name,
                'role': member.role,
                'capacity_percentage': member.capacity_percentage,
                'current_projects': len(member.current_projects or []),
                'productivity_score': member.productivity_score,
                'quality_score': member.quality_score
            }
        
        # Identify optimization opportunities
        overloaded_members = [m for m in team_members if m.capacity_percentage > 90]
        underutilized_members = [m for m in team_members if m.capacity_percentage < 60]
        
        return {
            'total_team_members': len(team_members),
            'average_capacity_utilization': np.mean([m.capacity_percentage for m in team_members]),
            'overloaded_members': len(overloaded_members),
            'underutilized_members': len(underutilized_members),
            'workload_distribution': workload_analysis,
            'rebalancing_recommendations': self._generate_rebalancing_recommendations(overloaded_members, underutilized_members)
        }
    
    def _generate_rebalancing_recommendations(self, overloaded: List[TeamMember], underutilized: List[TeamMember]) -> List[str]:
        """Generate workload rebalancing recommendations"""
        
        recommendations = []
        
        if overloaded:
            recommendations.append(f"Redistribute workload from {len(overloaded)} overloaded team members")
            recommendations.append("Consider bringing in freelance support for peak periods")
        
        if underutilized:
            recommendations.append(f"Increase project assignments for {len(underutilized)} underutilized team members")
            recommendations.append("Cross-train underutilized members for versatility")
        
        # Skill-based recommendations
        if overloaded and underutilized:
            recommendations.append("Match skill sets between overloaded and underutilized members for task delegation")
        
        return recommendations
    
    def _optimize_project_timelines(self, projects: List[Project]) -> Dict[str, Any]:
        """Optimize project timelines and identify conflicts"""
        
        timeline_conflicts = []
        optimization_opportunities = []
        
        # Sort projects by deadline
        sorted_projects = sorted(projects, key=lambda p: p.deadline or datetime.max.date())
        
        for i, project in enumerate(sorted_projects):
            # Check for timeline conflicts with other projects
            overlapping_projects = [p for p in sorted_projects[i+1:] if 
                                  p.deadline and project.deadline and 
                                  abs((p.deadline - project.deadline).days) < 7]
            
            if overlapping_projects:
                timeline_conflicts.append({
                    'project_id': project.project_id,
                    'project_name': project.project_name,
                    'deadline': project.deadline.isoformat(),
                    'conflicting_projects': len(overlapping_projects)
                })
            
            # Identify optimization opportunities
            if project.completion_percentage > 50 and project.risk_score < 30:
                optimization_opportunities.append({
                    'project_id': project.project_id,
                    'project_name': project.project_name,
                    'opportunity': 'early_completion_possible',
                    'potential_time_savings': '3-5 days'
                })
        
        return {
            'timeline_conflicts': timeline_conflicts,
            'optimization_opportunities': optimization_opportunities,
            'projects_at_risk': len([p for p in projects if p.risk_score > 60])
        }
    
    def optimize_brand_consistency(self, agency_id: str) -> Dict[str, Any]:
        """AI-powered brand consistency analysis and optimization"""
        
        # Get all creative assets
        assets = CreativeAsset.query.join(Project).filter(Project.agency_id == agency_id).all()
        
        if not assets:
            return {'error': 'No creative assets found'}
        
        # Group assets by client/project
        client_assets = {}
        for asset in assets:
            project = Project.query.filter_by(project_id=asset.project_id).first()
            if project:
                client_id = project.client_id
                if client_id not in client_assets:
                    client_assets[client_id] = []
                client_assets[client_id].append(asset)
        
        # Analyze brand consistency for each client
        consistency_analysis = []
        for client_id, client_asset_list in client_assets.items():
            client_analysis = self._analyze_client_brand_consistency(client_id, client_asset_list)
            consistency_analysis.append(client_analysis)
        
        # Generate overall recommendations
        recommendations = self._generate_brand_consistency_recommendations(consistency_analysis)
        
        return {
            'agency_id': agency_id,
            'clients_analyzed': len(client_assets),
            'total_assets_analyzed': len(assets),
            'brand_consistency_analysis': consistency_analysis,
            'recommendations': recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_client_brand_consistency(self, client_id: str, assets: List[CreativeAsset]) -> Dict[str, Any]:
        """Analyze brand consistency for a specific client"""
        
        client = Client.query.filter_by(client_id=client_id).first()
        if not client:
            return {'error': 'Client not found'}
        
        # Analyze color consistency
        color_consistency = self._analyze_color_consistency(assets, client.brand_guidelines)
        
        # Analyze typography consistency
        typography_consistency = self._analyze_typography_consistency(assets, client.brand_guidelines)
        
        # Calculate overall brand consistency score
        consistency_scores = [asset.brand_consistency_score for asset in assets if asset.brand_consistency_score]
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 75.0
        
        # Identify inconsistent assets
        inconsistent_assets = [asset for asset in assets if asset.brand_consistency_score < 70]
        
        return {
            'client_id': client_id,
            'client_name': client.company_name,
            'assets_analyzed': len(assets),
            'overall_consistency_score': overall_consistency,
            'color_consistency': color_consistency,
            'typography_consistency': typography_consistency,
            'inconsistent_assets_count': len(inconsistent_assets),
            'inconsistent_assets': [{
                'asset_id': asset.asset_id,
                'asset_name': asset.asset_name,
                'consistency_score': asset.brand_consistency_score,
                'issues': self._identify_consistency_issues(asset, client.brand_guidelines)
            } for asset in inconsistent_assets]
        }
    
    def _analyze_color_consistency(self, assets: List[CreativeAsset], brand_guidelines: Dict) -> Dict[str, Any]:
        """Analyze color palette consistency across assets"""
        
        brand_colors = brand_guidelines.get('colors', []) if brand_guidelines else []
        
        if not brand_colors:
            return {'status': 'no_brand_colors_defined'}
        
        # Analyze each asset's color usage
        color_compliance_scores = []
        for asset in assets:
            asset_colors = asset.color_palette or []
            
            # Calculate compliance based on brand color usage
            brand_color_usage = sum(1 for color in asset_colors if color in brand_colors)
            total_colors = len(asset_colors) if asset_colors else 1
            compliance_score = (brand_color_usage / total_colors * 100)
            color_compliance_scores.append(compliance_score)
        
        avg_compliance = np.mean(color_compliance_scores) if color_compliance_scores else 0
        
        return {
            'average_color_compliance': avg_compliance,
            'brand_colors_count': len(brand_colors),
            'assets_fully_compliant': len([score for score in color_compliance_scores if score >= 80]),
            'assets_needing_attention': len([score for score in color_compliance_scores if score < 60])
        }
    
    def _analyze_typography_consistency(self, assets: List[CreativeAsset], brand_guidelines: Dict) -> Dict[str, Any]:
        """Analyze typography consistency across assets"""
        
        brand_fonts = brand_guidelines.get('fonts', []) if brand_guidelines else []
        
        if not brand_fonts:
            return {'status': 'no_brand_fonts_defined'}
        
        # Analyze font usage across assets
        font_compliance_scores = []
        for asset in assets:
            asset_fonts = asset.fonts_used or []
            
            # Calculate compliance based on brand font usage
            brand_font_usage = sum(1 for font in asset_fonts if font in brand_fonts)
            total_fonts = len(asset_fonts) if asset_fonts else 1
            compliance_score = (brand_font_usage / total_fonts * 100)
            font_compliance_scores.append(compliance_score)
        
        avg_compliance = np.mean(font_compliance_scores) if font_compliance_scores else 0
        
        return {
            'average_font_compliance': avg_compliance,
            'brand_fonts_count': len(brand_fonts),
            'assets_fully_compliant': len([score for score in font_compliance_scores if score >= 80]),
            'assets_needing_attention': len([score for score in font_compliance_scores if score < 60])
        }
    
    def _identify_consistency_issues(self, asset: CreativeAsset, brand_guidelines: Dict) -> List[str]:
        """Identify specific brand consistency issues for an asset"""
        
        issues = []
        
        if brand_guidelines:
            # Color issues
            brand_colors = brand_guidelines.get('colors', [])
            asset_colors = asset.color_palette or []
            
            if brand_colors and asset_colors:
                non_brand_colors = [color for color in asset_colors if color not in brand_colors]
                if non_brand_colors:
                    issues.append(f"Uses non-brand colors: {', '.join(non_brand_colors[:3])}")
            
            # Font issues
            brand_fonts = brand_guidelines.get('fonts', [])
            asset_fonts = asset.fonts_used or []
            
            if brand_fonts and asset_fonts:
                non_brand_fonts = [font for font in asset_fonts if font not in brand_fonts]
                if non_brand_fonts:
                    issues.append(f"Uses non-brand fonts: {', '.join(non_brand_fonts[:2])}")
        
        # Quality issues
        if asset.brand_consistency_score < 50:
            issues.append("Significantly deviates from brand guidelines")
        
        return issues
    
    def _generate_brand_consistency_recommendations(self, consistency_analysis: List[Dict]) -> List[Dict[str, Any]]:
        """Generate brand consistency improvement recommendations"""
        
        recommendations = []
        
        # Analyze overall patterns
        low_consistency_clients = [analysis for analysis in consistency_analysis 
                                 if analysis.get('overall_consistency_score', 0) < 70]
        
        if low_consistency_clients:
            recommendations.append({
                'category': 'brand_compliance',
                'priority': 'high',
                'recommendation': f'Improve brand consistency for {len(low_consistency_clients)} clients with low compliance scores',
                'expected_benefit': 'Enhanced brand recognition and client satisfaction'
            })
        
        # Color consistency recommendations
        color_issues = sum(1 for analysis in consistency_analysis 
                          if analysis.get('color_consistency', {}).get('assets_needing_attention', 0) > 0)
        
        if color_issues > 0:
            recommendations.append({
                'category': 'color_compliance',
                'priority': 'medium',
                'recommendation': 'Implement automated color palette validation in design process',
                'expected_benefit': 'Consistent brand color usage across all assets'
            })
        
        # Typography recommendations
        font_issues = sum(1 for analysis in consistency_analysis 
                         if analysis.get('typography_consistency', {}).get('assets_needing_attention', 0) > 0)
        
        if font_issues > 0:
            recommendations.append({
                'category': 'typography_compliance',
                'priority': 'medium',
                'recommendation': 'Create typography templates and enforce brand font usage',
                'expected_benefit': 'Improved visual consistency and brand recognition'
            })
        
        # Process recommendations
        recommendations.append({
            'category': 'process_improvement',
            'priority': 'medium',
            'recommendation': 'Implement brand guideline checkpoints in creative review process',
            'expected_benefit': 'Prevent brand inconsistencies before final delivery'
        })
        
        return recommendations

# Initialize engine
creative_engine = CreativeIntelligenceEngine()

# Routes
@app.route('/creative-media')
def creative_dashboard():
    """Creative & Media Production dashboard"""
    
    recent_agencies = CreativeAgency.query.order_by(CreativeAgency.created_at.desc()).limit(10).all()
    
    return render_template('creative_media/dashboard.html',
                         agencies=recent_agencies)

@app.route('/creative-media/api/project-optimization', methods=['POST'])
def optimize_projects():
    """API endpoint for project optimization"""
    
    data = request.get_json()
    agency_id = data.get('agency_id')
    
    if not agency_id:
        return jsonify({'error': 'Agency ID required'}), 400
    
    optimization = creative_engine.analyze_project_optimization(agency_id)
    return jsonify(optimization)

@app.route('/creative-media/api/brand-consistency', methods=['POST'])
def analyze_brand_consistency():
    """API endpoint for brand consistency analysis"""
    
    data = request.get_json()
    agency_id = data.get('agency_id')
    
    if not agency_id:
        return jsonify({'error': 'Agency ID required'}), 400
    
    analysis = creative_engine.optimize_brand_consistency(agency_id)
    return jsonify(analysis)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if CreativeAgency.query.count() == 0:
        sample_agency = CreativeAgency(
            agency_id='CREATIVE_DEMO_001',
            agency_name='Demo Creative Studios',
            agency_type=AgencyType.MARKETING,
            location='Creative District',
            team_size=18,
            specializations=['brand_design', 'digital_marketing', 'video_production'],
            active_projects=12,
            monthly_revenue=185000,
            client_satisfaction_score=4.4
        )
        
        db.session.add(sample_agency)
        db.session.commit()
        logger.info("Sample creative media data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5028, debug=True)