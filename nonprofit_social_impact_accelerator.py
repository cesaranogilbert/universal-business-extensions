"""
Non-Profit & Social Impact Accelerator - Complete AI-Powered Social Good Platform
$12B+ Value Potential - Fundraising Optimization, Impact Measurement & Community Engagement
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "nonprofit-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///nonprofit.db")

db.init_app(app)

# Non-Profit Enums
class OrganizationType(Enum):
    CHARITY = "charity"
    NGO = "ngo"
    FOUNDATION = "foundation"
    SOCIAL_ENTERPRISE = "social_enterprise"
    COMMUNITY_ORG = "community_org"
    RELIGIOUS_ORG = "religious_org"

class CampaignType(Enum):
    FUNDRAISING = "fundraising"
    AWARENESS = "awareness"
    VOLUNTEER_RECRUITMENT = "volunteer_recruitment"
    ADVOCACY = "advocacy"
    EVENT = "event"

class DonorStatus(Enum):
    ACTIVE = "active"
    LAPSED = "lapsed"
    PROSPECT = "prospect"
    MAJOR_DONOR = "major_donor"

# Data Models
class NonProfitOrganization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_name = db.Column(db.String(200), nullable=False)
    organization_type = db.Column(db.Enum(OrganizationType), nullable=False)
    
    # Organization Details
    mission_statement = db.Column(db.Text)
    focus_areas = db.Column(db.JSON)  # List of cause areas
    geographic_scope = db.Column(db.String(100))  # local, national, international
    
    # Registration and Compliance
    tax_exempt_status = db.Column(db.String(50))  # 501c3, etc.
    registration_number = db.Column(db.String(100))
    
    # Organizational Metrics
    staff_count = db.Column(db.Integer, default=5)
    volunteer_count = db.Column(db.Integer, default=50)
    beneficiaries_served = db.Column(db.Integer, default=1000)
    
    # Financial Metrics
    annual_budget = db.Column(db.Float, default=250000.0)
    fundraising_goal = db.Column(db.Float, default=200000.0)
    program_expense_ratio = db.Column(db.Float, default=75.0)  # percentage to programs
    administrative_expense_ratio = db.Column(db.Float, default=15.0)  # percentage to admin
    
    # Impact Metrics
    impact_score = db.Column(db.Float, default=75.0)  # 0-100 overall impact rating
    transparency_score = db.Column(db.Float, default=85.0)  # 0-100 transparency rating
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Donor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    donor_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_id = db.Column(db.String(100), db.ForeignKey('non_profit_organization.organization_id'), nullable=False)
    
    # Personal Information
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    address = db.Column(db.Text)
    
    # Donor Classification
    donor_type = db.Column(db.String(50))  # individual, corporate, foundation, government
    status = db.Column(db.Enum(DonorStatus), default=DonorStatus.PROSPECT)
    
    # Giving History
    first_donation_date = db.Column(db.Date)
    last_donation_date = db.Column(db.Date)
    total_lifetime_giving = db.Column(db.Float, default=0.0)
    largest_single_gift = db.Column(db.Float, default=0.0)
    average_gift_size = db.Column(db.Float, default=0.0)
    donation_frequency = db.Column(db.Float, default=1.0)  # times per year
    
    # Engagement Metrics
    event_attendance_count = db.Column(db.Integer, default=0)
    volunteer_hours = db.Column(db.Float, default=0.0)
    newsletter_engagement = db.Column(db.Float, default=50.0)  # 0-100 engagement score
    social_media_engagement = db.Column(db.Float, default=25.0)  # 0-100 engagement score
    
    # Preferences
    preferred_communication = db.Column(db.String(50), default='email')
    giving_preferences = db.Column(db.JSON)  # Preferred causes, timing, amounts
    
    # AI Predictions
    lifetime_value_prediction = db.Column(db.Float, default=0.0)
    next_gift_prediction = db.Column(db.Float, default=0.0)
    churn_probability = db.Column(db.Float, default=0.25)  # 0-1
    optimal_ask_amount = db.Column(db.Float, default=100.0)
    
    # Wealth Indicators
    estimated_capacity = db.Column(db.Float, default=1000.0)
    philanthropic_interests = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Donation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    donation_id = db.Column(db.String(100), unique=True, nullable=False)
    donor_id = db.Column(db.String(100), db.ForeignKey('donor.donor_id'), nullable=False)
    organization_id = db.Column(db.String(100), db.ForeignKey('non_profit_organization.organization_id'), nullable=False)
    
    # Donation Details
    amount = db.Column(db.Float, nullable=False)
    donation_date = db.Column(db.Date, nullable=False)
    payment_method = db.Column(db.String(50))  # credit_card, check, cash, online, etc.
    
    # Campaign Information
    campaign_id = db.Column(db.String(100))
    appeal_source = db.Column(db.String(100))  # website, email, direct_mail, event, etc.
    
    # Donation Classification
    donation_type = db.Column(db.String(50))  # one_time, recurring, pledge, in_kind
    restriction = db.Column(db.String(100))  # unrestricted, program_specific, capital, etc.
    tribute_type = db.Column(db.String(50))  # memorial, honor, general
    
    # Processing Information
    processing_fee = db.Column(db.Float, default=0.0)
    net_amount = db.Column(db.Float)
    receipt_sent = db.Column(db.Boolean, default=False)
    thank_you_sent = db.Column(db.Boolean, default=False)
    
    # Tax Information
    tax_deductible_amount = db.Column(db.Float)
    receipt_number = db.Column(db.String(100))
    
    # Impact Tracking
    program_allocation = db.Column(db.JSON)  # How donation was allocated across programs
    impact_achieved = db.Column(db.Text)  # Specific impact from this donation
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Campaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    campaign_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_id = db.Column(db.String(100), db.ForeignKey('non_profit_organization.organization_id'), nullable=False)
    
    # Campaign Details
    campaign_name = db.Column(db.String(200), nullable=False)
    campaign_type = db.Column(db.Enum(CampaignType), nullable=False)
    description = db.Column(db.Text)
    
    # Timeline
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    
    # Goals and Targets
    fundraising_goal = db.Column(db.Float, default=0.0)
    donor_acquisition_goal = db.Column(db.Integer, default=0)
    engagement_goal = db.Column(db.Integer, default=0)
    
    # Performance Metrics
    total_raised = db.Column(db.Float, default=0.0)
    donor_count = db.Column(db.Integer, default=0)
    average_gift = db.Column(db.Float, default=0.0)
    conversion_rate = db.Column(db.Float, default=2.5)  # percentage
    
    # Outreach Metrics
    emails_sent = db.Column(db.Integer, default=0)
    social_media_reach = db.Column(db.Integer, default=0)
    website_visits = db.Column(db.Integer, default=0)
    
    # Costs
    campaign_budget = db.Column(db.Float, default=5000.0)
    actual_costs = db.Column(db.Float, default=0.0)
    cost_per_dollar_raised = db.Column(db.Float, default=0.15)
    
    # AI Optimization
    predicted_performance = db.Column(db.JSON)
    optimization_suggestions = db.Column(db.JSON)
    success_probability = db.Column(db.Float, default=75.0)  # percentage
    
    is_active = db.Column(db.Boolean, default=True)

class Volunteer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    volunteer_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_id = db.Column(db.String(100), db.ForeignKey('non_profit_organization.organization_id'), nullable=False)
    
    # Personal Information
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    age_range = db.Column(db.String(20))  # 18-25, 26-35, etc.
    
    # Volunteer Profile
    skills = db.Column(db.JSON)  # List of skills and expertise
    interests = db.Column(db.JSON)  # Areas of interest
    availability = db.Column(db.JSON)  # Days/times available
    
    # Engagement History
    start_date = db.Column(db.Date)
    total_hours_contributed = db.Column(db.Float, default=0.0)
    events_participated = db.Column(db.Integer, default=0)
    training_completed = db.Column(db.JSON)
    
    # Performance Metrics
    reliability_score = db.Column(db.Float, default=85.0)  # 0-100
    satisfaction_score = db.Column(db.Float, default=80.0)  # 0-100
    impact_rating = db.Column(db.Float, default=75.0)  # 0-100
    
    # Preferences
    preferred_roles = db.Column(db.JSON)
    communication_preferences = db.Column(db.JSON)
    
    # AI Insights
    retention_probability = db.Column(db.Float, default=70.0)  # percentage
    optimal_role_matches = db.Column(db.JSON)
    engagement_recommendations = db.Column(db.JSON)
    
    is_active = db.Column(db.Boolean, default=True)

class Program(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    program_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_id = db.Column(db.String(100), db.ForeignKey('non_profit_organization.organization_id'), nullable=False)
    
    # Program Details
    program_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    focus_area = db.Column(db.String(100))  # education, health, environment, etc.
    
    # Program Metrics
    beneficiaries_served = db.Column(db.Integer, default=0)
    program_budget = db.Column(db.Float, default=50000.0)
    cost_per_beneficiary = db.Column(db.Float, default=50.0)
    
    # Impact Measurement
    impact_metrics = db.Column(db.JSON)  # Key impact indicators
    outcomes_achieved = db.Column(db.JSON)  # Measured outcomes
    success_stories = db.Column(db.JSON)  # Collection of success stories
    
    # Efficiency Metrics
    efficiency_score = db.Column(db.Float, default=80.0)  # 0-100
    sustainability_score = db.Column(db.Float, default=75.0)  # 0-100
    innovation_score = db.Column(db.Float, default=70.0)  # 0-100
    
    # Funding
    funding_sources = db.Column(db.JSON)  # Grant, donation, government, etc.
    funding_sustainability = db.Column(db.Float, default=75.0)  # percentage sustainable
    
    # AI Analysis
    impact_prediction = db.Column(db.JSON)  # Predicted future impact
    optimization_opportunities = db.Column(db.JSON)
    scalability_assessment = db.Column(db.Float, default=65.0)  # 0-100
    
    start_date = db.Column(db.Date)
    is_active = db.Column(db.Boolean, default=True)

class ImpactMeasurement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    measurement_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_id = db.Column(db.String(100), db.ForeignKey('non_profit_organization.organization_id'), nullable=False)
    program_id = db.Column(db.String(100), db.ForeignKey('program.program_id'))
    
    # Measurement Details
    measurement_date = db.Column(db.Date, nullable=False)
    metric_name = db.Column(db.String(200), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    measurement_period = db.Column(db.String(50))  # monthly, quarterly, annually
    
    # Context
    target_value = db.Column(db.Float)
    baseline_value = db.Column(db.Float)
    measurement_method = db.Column(db.String(200))
    
    # Analysis
    progress_towards_goal = db.Column(db.Float, default=0.0)  # percentage
    trend_direction = db.Column(db.String(50))  # increasing, stable, decreasing
    
    # Validation
    data_quality_score = db.Column(db.Float, default=85.0)  # 0-100
    verification_status = db.Column(db.String(50), default='unverified')
    
    # AI Enhancement
    predicted_next_value = db.Column(db.Float, default=0.0)
    contributing_factors = db.Column(db.JSON)
    improvement_recommendations = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Non-Profit Intelligence Engine
class NonProfitIntelligenceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def optimize_fundraising_strategy(self, organization_id: str) -> Dict[str, Any]:
        """AI-powered fundraising optimization and donor development"""
        
        # Get donor data
        donors = Donor.query.filter_by(organization_id=organization_id).all()
        
        if not donors:
            return {'error': 'No donor data found'}
        
        # Analyze donor segments
        donor_segmentation = self._segment_donors(donors)
        
        # Predict donor behavior
        behavior_predictions = self._predict_donor_behavior(donors)
        
        # Optimize fundraising campaigns
        campaign_optimization = self._optimize_fundraising_campaigns(organization_id, donor_segmentation)
        
        # Calculate revenue potential
        revenue_potential = self._calculate_revenue_potential(donor_segmentation, behavior_predictions)
        
        return {
            'organization_id': organization_id,
            'total_donors': len(donors),
            'donor_segmentation': donor_segmentation,
            'behavior_predictions': behavior_predictions,
            'campaign_optimization': campaign_optimization,
            'revenue_potential': revenue_potential,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _segment_donors(self, donors: List[Donor]) -> Dict[str, Any]:
        """Segment donors based on giving patterns and engagement"""
        
        segments = {
            'major_donors': [],
            'regular_donors': [],
            'lapsed_donors': [],
            'prospects': [],
            'champions': []  # High engagement, advocates
        }
        
        # Calculate thresholds
        total_giving = [d.total_lifetime_giving for d in donors]
        major_donor_threshold = np.percentile(total_giving, 90) if total_giving else 1000
        
        for donor in donors:
            # Major donors (top 10% by lifetime giving)
            if donor.total_lifetime_giving >= major_donor_threshold:
                segments['major_donors'].append(self._create_donor_summary(donor))
            
            # Lapsed donors (no donation in 18+ months)
            elif donor.last_donation_date and (datetime.now().date() - donor.last_donation_date).days > 540:
                segments['lapsed_donors'].append(self._create_donor_summary(donor))
            
            # Champions (high engagement across multiple channels)
            elif (donor.volunteer_hours > 20 or donor.event_attendance_count > 3 or 
                  donor.social_media_engagement > 70):
                segments['champions'].append(self._create_donor_summary(donor))
            
            # Regular donors
            elif donor.total_lifetime_giving > 0 and donor.donation_frequency >= 1:
                segments['regular_donors'].append(self._create_donor_summary(donor))
            
            # Prospects
            else:
                segments['prospects'].append(self._create_donor_summary(donor))
        
        # Calculate segment statistics
        segment_stats = {}
        for segment_name, segment_donors in segments.items():
            if segment_donors:
                total_value = sum(d['lifetime_giving'] for d in segment_donors)
                avg_gift = np.mean([d['average_gift'] for d in segment_donors])
                
                segment_stats[segment_name] = {
                    'count': len(segment_donors),
                    'total_lifetime_value': total_value,
                    'average_gift_size': avg_gift,
                    'percentage_of_total': len(segment_donors) / len(donors) * 100
                }
            else:
                segment_stats[segment_name] = {
                    'count': 0,
                    'total_lifetime_value': 0,
                    'average_gift_size': 0,
                    'percentage_of_total': 0
                }
        
        return {
            'segments': segments,
            'segment_statistics': segment_stats
        }
    
    def _create_donor_summary(self, donor: Donor) -> Dict[str, Any]:
        """Create a summary of donor for segmentation"""
        
        return {
            'donor_id': donor.donor_id,
            'name': donor.name,
            'lifetime_giving': donor.total_lifetime_giving,
            'average_gift': donor.average_gift_size,
            'last_gift_date': donor.last_donation_date.isoformat() if donor.last_donation_date else None,
            'engagement_score': (donor.newsletter_engagement + donor.social_media_engagement) / 2,
            'predicted_next_gift': donor.next_gift_prediction,
            'churn_probability': donor.churn_probability
        }
    
    def _predict_donor_behavior(self, donors: List[Donor]) -> Dict[str, Any]:
        """Predict donor behavior and giving patterns"""
        
        # Retention analysis
        at_risk_donors = [d for d in donors if d.churn_probability > 0.6]
        likely_to_upgrade = [d for d in donors if d.next_gift_prediction > d.average_gift_size * 1.5]
        
        # Lifetime value predictions
        total_predicted_ltv = sum(d.lifetime_value_prediction for d in donors)
        
        # Engagement predictions
        high_engagement_potential = [d for d in donors if 
                                   d.newsletter_engagement > 60 and d.volunteer_hours == 0]
        
        return {
            'retention_analysis': {
                'at_risk_donors': len(at_risk_donors),
                'stable_donors': len([d for d in donors if d.churn_probability < 0.3]),
                'retention_rate_prediction': len([d for d in donors if d.churn_probability < 0.5]) / len(donors) * 100
            },
            'upgrade_potential': {
                'likely_to_upgrade': len(likely_to_upgrade),
                'potential_additional_revenue': sum(d.next_gift_prediction - d.average_gift_size 
                                                  for d in likely_to_upgrade if d.next_gift_prediction > d.average_gift_size)
            },
            'lifetime_value': {
                'total_predicted_ltv': total_predicted_ltv,
                'average_predicted_ltv': total_predicted_ltv / len(donors) if donors else 0
            },
            'engagement_opportunities': {
                'volunteer_recruitment_potential': len(high_engagement_potential),
                'advocacy_potential': len([d for d in donors if d.social_media_engagement > 70])
            }
        }
    
    def _optimize_fundraising_campaigns(self, organization_id: str, donor_segmentation: Dict) -> Dict[str, Any]:
        """Optimize fundraising campaigns based on donor analysis"""
        
        # Get recent campaign performance
        recent_campaigns = Campaign.query.filter_by(organization_id=organization_id)\
                                        .filter(Campaign.start_date >= datetime.now().date() - timedelta(days=365))\
                                        .all()
        
        campaign_recommendations = []
        
        # Major donor campaign
        major_donors = donor_segmentation['segments']['major_donors']
        if major_donors:
            campaign_recommendations.append({
                'campaign_type': 'major_donor_cultivation',
                'target_segment': 'major_donors',
                'target_count': len(major_donors),
                'recommended_approach': 'Personal meetings and exclusive events',
                'expected_response_rate': '60-80%',
                'potential_revenue': sum(d['lifetime_giving'] for d in major_donors) * 0.3
            })
        
        # Lapsed donor reactivation
        lapsed_donors = donor_segmentation['segments']['lapsed_donors']
        if lapsed_donors:
            campaign_recommendations.append({
                'campaign_type': 'win_back_campaign',
                'target_segment': 'lapsed_donors',
                'target_count': len(lapsed_donors),
                'recommended_approach': 'Special "We miss you" offer with impact stories',
                'expected_response_rate': '15-25%',
                'potential_revenue': len(lapsed_donors) * 75  # Conservative estimate
            })
        
        # Prospect acquisition
        prospects = donor_segmentation['segments']['prospects']
        if prospects:
            campaign_recommendations.append({
                'campaign_type': 'acquisition_campaign',
                'target_segment': 'prospects',
                'target_count': len(prospects),
                'recommended_approach': 'Multi-channel awareness and first-gift incentives',
                'expected_response_rate': '5-10%',
                'potential_revenue': len(prospects) * 0.08 * 50  # 8% response, $50 average
            })
        
        # Calculate campaign performance benchmarks
        if recent_campaigns:
            avg_response_rate = np.mean([c.conversion_rate for c in recent_campaigns])
            avg_cost_per_dollar = np.mean([c.cost_per_dollar_raised for c in recent_campaigns])
            
            performance_benchmarks = {
                'average_response_rate': avg_response_rate,
                'average_cost_per_dollar_raised': avg_cost_per_dollar,
                'campaigns_analyzed': len(recent_campaigns)
            }
        else:
            performance_benchmarks = {
                'average_response_rate': 0,
                'average_cost_per_dollar_raised': 0,
                'campaigns_analyzed': 0
            }
        
        return {
            'campaign_recommendations': campaign_recommendations,
            'performance_benchmarks': performance_benchmarks,
            'optimization_priorities': [
                'Focus on major donor relationship building',
                'Implement systematic lapsed donor reactivation',
                'Develop multi-touch prospect nurturing sequence'
            ]
        }
    
    def _calculate_revenue_potential(self, donor_segmentation: Dict, behavior_predictions: Dict) -> Dict[str, Any]:
        """Calculate potential revenue increases from optimization"""
        
        # Current annual giving estimate
        segment_stats = donor_segmentation['segment_statistics']
        current_annual_estimate = sum(stats['total_lifetime_value'] for stats in segment_stats.values()) * 0.3
        
        # Potential increases
        major_donor_potential = segment_stats['major_donors']['total_lifetime_value'] * 0.15  # 15% increase
        retention_savings = behavior_predictions['retention_analysis']['at_risk_donors'] * 150  # Avg donor value
        upgrade_potential = behavior_predictions['upgrade_potential']['potential_additional_revenue']
        
        total_potential_increase = major_donor_potential + retention_savings + upgrade_potential
        
        return {
            'current_estimated_annual_giving': current_annual_estimate,
            'potential_revenue_increases': {
                'major_donor_cultivation': major_donor_potential,
                'retention_improvement': retention_savings,
                'donor_upgrades': upgrade_potential
            },
            'total_potential_increase': total_potential_increase,
            'percentage_increase': (total_potential_increase / current_annual_estimate * 100) if current_annual_estimate > 0 else 0,
            'implementation_timeline': '6-12 months for full realization'
        }
    
    def measure_social_impact(self, organization_id: str) -> Dict[str, Any]:
        """AI-powered social impact measurement and optimization"""
        
        # Get programs and impact data
        programs = Program.query.filter_by(organization_id=organization_id, is_active=True).all()
        impact_measurements = ImpactMeasurement.query.filter_by(organization_id=organization_id)\
                                                    .filter(ImpactMeasurement.measurement_date >= datetime.now().date() - timedelta(days=365))\
                                                    .all()
        
        if not programs:
            return {'error': 'No active programs found'}
        
        # Analyze program effectiveness
        program_analysis = []
        for program in programs:
            program_impact = self._analyze_program_impact(program, impact_measurements)
            program_analysis.append(program_impact)
        
        # Calculate overall impact metrics
        overall_impact = self._calculate_overall_impact(programs, impact_measurements)
        
        # Generate improvement recommendations
        recommendations = self._generate_impact_recommendations(program_analysis, overall_impact)
        
        return {
            'organization_id': organization_id,
            'programs_analyzed': len(programs),
            'measurement_period': '12 months',
            'program_impact_analysis': program_analysis,
            'overall_impact_metrics': overall_impact,
            'improvement_recommendations': recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_program_impact(self, program: Program, measurements: List[ImpactMeasurement]) -> Dict[str, Any]:
        """Analyze impact for individual program"""
        
        # Get measurements for this program
        program_measurements = [m for m in measurements if m.program_id == program.program_id]
        
        if not program_measurements:
            return {
                'program_id': program.program_id,
                'program_name': program.program_name,
                'status': 'no_measurements_available'
            }
        
        # Calculate impact metrics
        impact_trends = self._calculate_impact_trends(program_measurements)
        efficiency_analysis = self._analyze_program_efficiency(program, program_measurements)
        
        # Cost-effectiveness analysis
        cost_per_beneficiary = program.cost_per_beneficiary
        cost_effectiveness_score = self._calculate_cost_effectiveness(program, program_measurements)
        
        return {
            'program_id': program.program_id,
            'program_name': program.program_name,
            'focus_area': program.focus_area,
            'beneficiaries_served': program.beneficiaries_served,
            'program_budget': program.program_budget,
            'cost_per_beneficiary': cost_per_beneficiary,
            'impact_trends': impact_trends,
            'efficiency_analysis': efficiency_analysis,
            'cost_effectiveness_score': cost_effectiveness_score,
            'overall_program_score': (program.efficiency_score + program.sustainability_score + cost_effectiveness_score) / 3
        }
    
    def _calculate_impact_trends(self, measurements: List[ImpactMeasurement]) -> Dict[str, Any]:
        """Calculate impact trends from measurements"""
        
        if len(measurements) < 2:
            return {'status': 'insufficient_data_for_trends'}
        
        # Sort by date
        sorted_measurements = sorted(measurements, key=lambda m: m.measurement_date)
        
        # Calculate trend for each metric
        metric_trends = {}
        for metric_name in set(m.metric_name for m in measurements):
            metric_values = [(m.measurement_date, m.metric_value) for m in sorted_measurements if m.metric_name == metric_name]
            
            if len(metric_values) >= 2:
                # Simple trend calculation
                first_value = metric_values[0][1]
                last_value = metric_values[-1][1]
                
                if first_value > 0:
                    percent_change = ((last_value - first_value) / first_value) * 100
                    trend_direction = 'increasing' if percent_change > 5 else ('decreasing' if percent_change < -5 else 'stable')
                else:
                    percent_change = 0
                    trend_direction = 'stable'
                
                metric_trends[metric_name] = {
                    'percent_change': percent_change,
                    'trend_direction': trend_direction,
                    'current_value': last_value,
                    'measurement_count': len(metric_values)
                }
        
        return metric_trends
    
    def _analyze_program_efficiency(self, program: Program, measurements: List[ImpactMeasurement]) -> Dict[str, Any]:
        """Analyze program operational efficiency"""
        
        # Calculate actual vs target performance
        target_achievements = []
        for measurement in measurements:
            if measurement.target_value and measurement.target_value > 0:
                achievement_rate = (measurement.metric_value / measurement.target_value) * 100
                target_achievements.append(achievement_rate)
        
        avg_target_achievement = np.mean(target_achievements) if target_achievements else 0
        
        # Budget efficiency
        budget_per_beneficiary = program.program_budget / program.beneficiaries_served if program.beneficiaries_served > 0 else 0
        
        # Compare to sector benchmarks (simplified)
        sector_benchmark_efficiency = 75.0  # Assumed benchmark
        relative_efficiency = (program.efficiency_score / sector_benchmark_efficiency) * 100
        
        return {
            'target_achievement_rate': avg_target_achievement,
            'budget_efficiency': budget_per_beneficiary,
            'relative_to_sector': relative_efficiency,
            'efficiency_score': program.efficiency_score,
            'improvement_potential': max(0, 90 - program.efficiency_score)
        }
    
    def _calculate_cost_effectiveness(self, program: Program, measurements: List[ImpactMeasurement]) -> float:
        """Calculate cost-effectiveness score for program"""
        
        # Get impact measurements
        if not measurements:
            return 50.0  # Default score
        
        # Calculate impact per dollar spent
        total_impact_value = sum(m.metric_value for m in measurements if m.metric_value > 0)
        cost_per_impact = program.program_budget / total_impact_value if total_impact_value > 0 else 0
        
        # Score based on cost efficiency (lower cost per impact = higher score)
        if cost_per_impact > 0:
            # Normalized score (simplified)
            cost_effectiveness_score = min(100, max(20, 100 - (cost_per_impact / 100)))
        else:
            cost_effectiveness_score = 50.0
        
        return cost_effectiveness_score
    
    def _calculate_overall_impact(self, programs: List[Program], measurements: List[ImpactMeasurement]) -> Dict[str, Any]:
        """Calculate organization-wide impact metrics"""
        
        # Aggregate beneficiaries
        total_beneficiaries = sum(p.beneficiaries_served for p in programs)
        total_budget = sum(p.program_budget for p in programs)
        
        # Impact efficiency
        overall_efficiency = np.mean([p.efficiency_score for p in programs])
        overall_sustainability = np.mean([p.sustainability_score for p in programs])
        
        # Measurement quality
        measurements_with_targets = len([m for m in measurements if m.target_value])
        measurement_quality = (measurements_with_targets / len(measurements) * 100) if measurements else 0
        
        # Calculate impact score
        impact_score = (overall_efficiency + overall_sustainability + measurement_quality) / 3
        
        return {
            'total_beneficiaries_served': total_beneficiaries,
            'total_program_budget': total_budget,
            'cost_per_beneficiary_overall': total_budget / total_beneficiaries if total_beneficiaries > 0 else 0,
            'overall_efficiency_score': overall_efficiency,
            'overall_sustainability_score': overall_sustainability,
            'measurement_quality_score': measurement_quality,
            'composite_impact_score': impact_score,
            'active_programs': len(programs),
            'total_measurements': len(measurements)
        }
    
    def _generate_impact_recommendations(self, program_analysis: List[Dict], overall_impact: Dict) -> List[Dict[str, Any]]:
        """Generate recommendations for impact improvement"""
        
        recommendations = []
        
        # Program-specific recommendations
        low_performing_programs = [p for p in program_analysis if p.get('overall_program_score', 0) < 70]
        if low_performing_programs:
            recommendations.append({
                'category': 'program_improvement',
                'priority': 'high',
                'recommendation': f'Focus improvement efforts on {len(low_performing_programs)} underperforming programs',
                'expected_impact': 'Increase overall impact effectiveness by 20-30%'
            })
        
        # Measurement improvements
        if overall_impact['measurement_quality_score'] < 80:
            recommendations.append({
                'category': 'measurement_enhancement',
                'priority': 'medium',
                'recommendation': 'Improve impact measurement consistency and target setting',
                'expected_impact': 'Better tracking and demonstration of results'
            })
        
        # Cost efficiency
        high_cost_programs = [p for p in program_analysis if p.get('cost_per_beneficiary', 0) > overall_impact.get('cost_per_beneficiary_overall', 0) * 1.5]
        if high_cost_programs:
            recommendations.append({
                'category': 'cost_efficiency',
                'priority': 'medium',
                'recommendation': f'Review and optimize {len(high_cost_programs)} high-cost programs',
                'expected_impact': 'Reduce costs while maintaining service quality'
            })
        
        # Sustainability
        if overall_impact['overall_sustainability_score'] < 75:
            recommendations.append({
                'category': 'sustainability',
                'priority': 'high',
                'recommendation': 'Develop more sustainable funding models and program designs',
                'expected_impact': 'Ensure long-term program viability and impact'
            })
        
        return recommendations

# Initialize engine
nonprofit_engine = NonProfitIntelligenceEngine()

# Routes
@app.route('/nonprofit')
def nonprofit_dashboard():
    """Non-Profit & Social Impact dashboard"""
    
    recent_organizations = NonProfitOrganization.query.order_by(NonProfitOrganization.created_at.desc()).limit(10).all()
    
    return render_template('nonprofit/dashboard.html',
                         organizations=recent_organizations)

@app.route('/nonprofit/api/fundraising-optimization', methods=['POST'])
def optimize_fundraising():
    """API endpoint for fundraising optimization"""
    
    data = request.get_json()
    organization_id = data.get('organization_id')
    
    if not organization_id:
        return jsonify({'error': 'Organization ID required'}), 400
    
    optimization = nonprofit_engine.optimize_fundraising_strategy(organization_id)
    return jsonify(optimization)

@app.route('/nonprofit/api/impact-measurement', methods=['POST'])
def measure_impact():
    """API endpoint for social impact measurement"""
    
    data = request.get_json()
    organization_id = data.get('organization_id')
    
    if not organization_id:
        return jsonify({'error': 'Organization ID required'}), 400
    
    measurement = nonprofit_engine.measure_social_impact(organization_id)
    return jsonify(measurement)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if NonProfitOrganization.query.count() == 0:
        sample_org = NonProfitOrganization(
            organization_id='NPO_DEMO_001',
            organization_name='Demo Community Foundation',
            organization_type=OrganizationType.FOUNDATION,
            mission_statement='Empowering communities through education and social services',
            focus_areas=['education', 'community_development', 'poverty_alleviation'],
            geographic_scope='regional',
            staff_count=12,
            volunteer_count=150,
            beneficiaries_served=2500,
            annual_budget=850000,
            impact_score=82.5
        )
        
        db.session.add(sample_org)
        db.session.commit()
        logger.info("Sample non-profit data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5029, debug=True)