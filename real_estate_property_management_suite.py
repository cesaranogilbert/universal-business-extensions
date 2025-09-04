"""
Real Estate & Property Management Suite - Complete AI-Powered Real Estate Platform
$20B+ Value Potential - Property Valuation, Tenant Management & Market Intelligence
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
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "real-estate-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///real_estate.db")

db.init_app(app)

# Real Estate Enums
class PropertyType(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    RETAIL = "retail"
    MIXED_USE = "mixed_use"

class PropertyStatus(Enum):
    AVAILABLE = "available"
    RENTED = "rented"
    MAINTENANCE = "maintenance"
    VACANT = "vacant"

class TenantStatus(Enum):
    ACTIVE = "active"
    PENDING = "pending"
    TERMINATED = "terminated"
    DELINQUENT = "delinquent"

# Data Models
class RealEstateAgency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agency_id = db.Column(db.String(100), unique=True, nullable=False)
    agency_name = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200))
    
    # Agency Configuration
    property_count = db.Column(db.Integer, default=0)
    agent_count = db.Column(db.Integer, default=1)
    specializations = db.Column(db.JSON)  # residential, commercial, etc.
    
    # Performance Metrics
    monthly_revenue = db.Column(db.Float, default=0.0)
    average_sale_price = db.Column(db.Float, default=0.0)
    average_days_on_market = db.Column(db.Integer, default=45)
    commission_rate = db.Column(db.Float, default=6.0)  # percentage
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Property(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.String(100), unique=True, nullable=False)
    agency_id = db.Column(db.String(100), db.ForeignKey('real_estate_agency.agency_id'), nullable=False)
    
    # Property Details
    address = db.Column(db.String(500), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    zip_code = db.Column(db.String(20))
    property_type = db.Column(db.Enum(PropertyType), nullable=False)
    
    # Physical Characteristics
    square_footage = db.Column(db.Integer)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    lot_size = db.Column(db.Float)  # acres
    year_built = db.Column(db.Integer)
    
    # Financial Information
    current_value = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float)
    monthly_rent = db.Column(db.Float, default=0.0)
    property_taxes = db.Column(db.Float, default=0.0)
    insurance_cost = db.Column(db.Float, default=0.0)
    maintenance_cost = db.Column(db.Float, default=0.0)
    
    # Market Analysis
    comparable_properties = db.Column(db.JSON)  # nearby similar properties
    market_value_estimate = db.Column(db.Float, default=0.0)
    appreciation_rate = db.Column(db.Float, default=3.0)  # annual percentage
    
    # Rental Information
    status = db.Column(db.Enum(PropertyStatus), default=PropertyStatus.AVAILABLE)
    rental_yield = db.Column(db.Float, default=0.0)  # annual percentage
    occupancy_rate = db.Column(db.Float, default=100.0)  # percentage
    
    # AI Predictions
    predicted_value_6_months = db.Column(db.Float, default=0.0)
    predicted_rent_6_months = db.Column(db.Float, default=0.0)
    investment_score = db.Column(db.Float, default=75.0)  # 0-100
    
    # Marketing
    listing_date = db.Column(db.Date)
    days_on_market = db.Column(db.Integer, default=0)
    view_count = db.Column(db.Integer, default=0)
    inquiry_count = db.Column(db.Integer, default=0)
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Tenant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.String(100), unique=True, nullable=False)
    property_id = db.Column(db.String(100), db.ForeignKey('property.property_id'), nullable=False)
    
    # Tenant Details
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    emergency_contact = db.Column(db.JSON)
    
    # Lease Information
    lease_start_date = db.Column(db.Date, nullable=False)
    lease_end_date = db.Column(db.Date, nullable=False)
    monthly_rent = db.Column(db.Float, nullable=False)
    security_deposit = db.Column(db.Float, default=0.0)
    
    # Status and Performance
    status = db.Column(db.Enum(TenantStatus), default=TenantStatus.ACTIVE)
    payment_history = db.Column(db.JSON)  # payment records
    late_payments = db.Column(db.Integer, default=0)
    total_rent_paid = db.Column(db.Float, default=0.0)
    
    # Screening Information
    credit_score = db.Column(db.Integer)
    annual_income = db.Column(db.Float)
    employment_verification = db.Column(db.Boolean, default=False)
    references = db.Column(db.JSON)
    
    # AI Analysis
    risk_score = db.Column(db.Float, default=25.0)  # 0-100, higher = higher risk
    renewal_probability = db.Column(db.Float, default=70.0)  # percentage
    recommended_rent_increase = db.Column(db.Float, default=0.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MaintenanceRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(100), unique=True, nullable=False)
    property_id = db.Column(db.String(100), db.ForeignKey('property.property_id'), nullable=False)
    tenant_id = db.Column(db.String(100), db.ForeignKey('tenant.tenant_id'))
    
    # Request Details
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100))  # plumbing, electrical, hvac, etc.
    priority = db.Column(db.String(20), default='medium')  # low, medium, high, emergency
    
    # Status and Timeline
    status = db.Column(db.String(50), default='open')  # open, in_progress, completed, cancelled
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    assigned_date = db.Column(db.DateTime)
    completed_date = db.Column(db.DateTime)
    
    # Service Provider
    assigned_contractor = db.Column(db.String(200))
    contractor_contact = db.Column(db.String(200))
    estimated_cost = db.Column(db.Float, default=0.0)
    actual_cost = db.Column(db.Float, default=0.0)
    
    # Quality and Satisfaction
    tenant_satisfaction = db.Column(db.Float, default=0.0)  # 1-5 stars
    work_quality_rating = db.Column(db.Float, default=0.0)  # 1-5 stars
    
    # AI Analysis
    cost_prediction = db.Column(db.Float, default=0.0)
    urgency_score = db.Column(db.Float, default=50.0)  # 0-100
    recommended_contractor = db.Column(db.String(200))

class MarketData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    market_id = db.Column(db.String(100), unique=True, nullable=False)
    
    # Location Information
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    zip_code = db.Column(db.String(20))
    neighborhood = db.Column(db.String(100))
    
    # Market Metrics
    date = db.Column(db.Date, nullable=False)
    median_home_price = db.Column(db.Float, default=0.0)
    median_rent = db.Column(db.Float, default=0.0)
    price_per_sqft = db.Column(db.Float, default=0.0)
    days_on_market = db.Column(db.Integer, default=0)
    
    # Market Trends
    price_appreciation_12m = db.Column(db.Float, default=0.0)  # percentage
    rent_growth_12m = db.Column(db.Float, default=0.0)  # percentage
    inventory_months = db.Column(db.Float, default=6.0)  # months of supply
    
    # Economic Indicators
    unemployment_rate = db.Column(db.Float, default=0.0)
    population_growth = db.Column(db.Float, default=0.0)
    median_income = db.Column(db.Float, default=0.0)
    
    # AI Predictions
    predicted_price_change_6m = db.Column(db.Float, default=0.0)
    predicted_rent_change_6m = db.Column(db.Float, default=0.0)
    market_score = db.Column(db.Float, default=75.0)  # 0-100
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Real Estate Intelligence Engine
class RealEstateIntelligenceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def analyze_property_valuation(self, property_id: str) -> Dict[str, Any]:
        """AI-powered property valuation analysis"""
        
        property_obj = Property.query.filter_by(property_id=property_id).first()
        if not property_obj:
            return {'error': 'Property not found'}
        
        # Get comparable properties
        comparables = self._find_comparable_properties(property_obj)
        
        # Analyze market data
        market_analysis = self._analyze_local_market(property_obj)
        
        # Calculate valuation
        valuation_analysis = self._calculate_property_valuation(property_obj, comparables, market_analysis)
        
        # Generate investment analysis
        investment_analysis = self._analyze_investment_potential(property_obj, valuation_analysis)
        
        return {
            'property_id': property_id,
            'current_value': property_obj.current_value,
            'valuation_analysis': valuation_analysis,
            'comparable_properties': comparables,
            'market_analysis': market_analysis,
            'investment_analysis': investment_analysis,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _find_comparable_properties(self, property_obj: Property) -> List[Dict[str, Any]]:
        """Find comparable properties for valuation"""
        
        # Search for similar properties in the same area
        similar_properties = Property.query.filter(
            Property.city == property_obj.city,
            Property.property_type == property_obj.property_type,
            Property.property_id != property_obj.property_id
        ).limit(10).all()
        
        comparables = []
        for prop in similar_properties:
            # Calculate similarity score
            similarity_score = self._calculate_property_similarity(property_obj, prop)
            
            if similarity_score > 0.6:  # 60% similarity threshold
                comparables.append({
                    'property_id': prop.property_id,
                    'address': prop.address,
                    'square_footage': prop.square_footage,
                    'bedrooms': prop.bedrooms,
                    'bathrooms': prop.bathrooms,
                    'current_value': prop.current_value,
                    'price_per_sqft': prop.current_value / prop.square_footage if prop.square_footage else 0,
                    'similarity_score': similarity_score,
                    'days_on_market': prop.days_on_market
                })
        
        # Sort by similarity score
        return sorted(comparables, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    def _calculate_property_similarity(self, prop1: Property, prop2: Property) -> float:
        """Calculate similarity score between two properties"""
        
        similarity_factors = []
        
        # Size similarity
        if prop1.square_footage and prop2.square_footage:
            size_diff = abs(prop1.square_footage - prop2.square_footage) / max(prop1.square_footage, prop2.square_footage)
            size_similarity = max(0, 1 - size_diff)
            similarity_factors.append(size_similarity * 0.3)
        
        # Bedroom similarity
        if prop1.bedrooms and prop2.bedrooms:
            bedroom_similarity = 1.0 if prop1.bedrooms == prop2.bedrooms else 0.7
            similarity_factors.append(bedroom_similarity * 0.2)
        
        # Age similarity
        if prop1.year_built and prop2.year_built:
            age_diff = abs(prop1.year_built - prop2.year_built) / 50  # Normalize by 50 years
            age_similarity = max(0, 1 - age_diff)
            similarity_factors.append(age_similarity * 0.2)
        
        # Location proximity (simplified - same city gets 0.3)
        if prop1.city == prop2.city:
            similarity_factors.append(0.3)
        
        return sum(similarity_factors) if similarity_factors else 0.0
    
    def _analyze_local_market(self, property_obj: Property) -> Dict[str, Any]:
        """Analyze local market conditions"""
        
        # Get recent market data for the area
        market_data = MarketData.query.filter_by(
            city=property_obj.city,
            state=property_obj.state
        ).order_by(MarketData.date.desc()).first()
        
        if not market_data:
            return {'error': 'No market data available'}
        
        # Calculate market trends
        market_trend = 'stable'
        if market_data.price_appreciation_12m > 5:
            market_trend = 'appreciating'
        elif market_data.price_appreciation_12m < -2:
            market_trend = 'declining'
        
        # Market strength indicators
        market_strength = 'balanced'
        if market_data.days_on_market < 30 and market_data.inventory_months < 4:
            market_strength = 'seller_market'
        elif market_data.days_on_market > 60 and market_data.inventory_months > 8:
            market_strength = 'buyer_market'
        
        return {
            'median_home_price': market_data.median_home_price,
            'median_price_per_sqft': market_data.price_per_sqft,
            'price_appreciation_12m': market_data.price_appreciation_12m,
            'days_on_market': market_data.days_on_market,
            'inventory_months': market_data.inventory_months,
            'market_trend': market_trend,
            'market_strength': market_strength,
            'predicted_price_change_6m': market_data.predicted_price_change_6m
        }
    
    def _calculate_property_valuation(self, property_obj: Property, comparables: List[Dict], market_analysis: Dict) -> Dict[str, Any]:
        """Calculate comprehensive property valuation"""
        
        valuation_methods = {}
        
        # Comparative Market Analysis (CMA)
        if comparables:
            comparable_prices = [comp['price_per_sqft'] for comp in comparables if comp['price_per_sqft'] > 0]
            if comparable_prices:
                avg_price_per_sqft = np.mean(comparable_prices)
                cma_value = avg_price_per_sqft * property_obj.square_footage if property_obj.square_footage else 0
                valuation_methods['comparative_market_analysis'] = cma_value
        
        # Market-based valuation
        if 'median_price_per_sqft' in market_analysis and property_obj.square_footage:
            market_value = market_analysis['median_price_per_sqft'] * property_obj.square_footage
            valuation_methods['market_based'] = market_value
        
        # Income approach (for rental properties)
        if property_obj.monthly_rent > 0:
            annual_income = property_obj.monthly_rent * 12
            # Use cap rate of 8% as default
            cap_rate = 0.08
            income_value = annual_income / cap_rate
            valuation_methods['income_approach'] = income_value
        
        # Calculate weighted average
        if valuation_methods:
            # Equal weighting for now, could be sophisticated based on property type
            estimated_value = np.mean(list(valuation_methods.values()))
        else:
            estimated_value = property_obj.current_value
        
        # Calculate confidence score
        confidence_score = min(100, len(comparables) * 20 + 40)  # Higher with more comparables
        
        return {
            'estimated_value': estimated_value,
            'current_listed_value': property_obj.current_value,
            'value_difference': estimated_value - property_obj.current_value,
            'value_difference_percent': ((estimated_value - property_obj.current_value) / property_obj.current_value * 100) if property_obj.current_value > 0 else 0,
            'valuation_methods': valuation_methods,
            'confidence_score': confidence_score,
            'valuation_range': {
                'low': estimated_value * 0.9,
                'high': estimated_value * 1.1
            }
        }
    
    def _analyze_investment_potential(self, property_obj: Property, valuation_analysis: Dict) -> Dict[str, Any]:
        """Analyze investment potential of property"""
        
        # Calculate key investment metrics
        annual_rent = property_obj.monthly_rent * 12 if property_obj.monthly_rent else 0
        annual_expenses = (property_obj.property_taxes + property_obj.insurance_cost + property_obj.maintenance_cost) * 12
        net_operating_income = annual_rent - annual_expenses
        
        # Cap rate
        cap_rate = (net_operating_income / property_obj.current_value * 100) if property_obj.current_value > 0 else 0
        
        # Cash-on-cash return (assuming 20% down payment)
        down_payment = property_obj.current_value * 0.2
        annual_debt_service = property_obj.current_value * 0.8 * 0.06  # Assuming 6% mortgage rate
        cash_flow = net_operating_income - annual_debt_service
        cash_on_cash_return = (cash_flow / down_payment * 100) if down_payment > 0 else 0
        
        # Investment score calculation
        investment_score = 50  # Base score
        
        if cap_rate > 8:
            investment_score += 20
        elif cap_rate > 6:
            investment_score += 10
        
        if cash_on_cash_return > 12:
            investment_score += 20
        elif cash_on_cash_return > 8:
            investment_score += 10
        
        if valuation_analysis['value_difference_percent'] < -10:  # Undervalued
            investment_score += 15
        
        investment_score = min(100, max(0, investment_score))
        
        return {
            'cap_rate': cap_rate,
            'cash_on_cash_return': cash_on_cash_return,
            'net_operating_income': net_operating_income,
            'annual_cash_flow': cash_flow,
            'investment_score': investment_score,
            'investment_recommendation': self._get_investment_recommendation(investment_score),
            'risk_factors': self._identify_investment_risks(property_obj),
            'opportunities': self._identify_investment_opportunities(property_obj, valuation_analysis)
        }
    
    def _get_investment_recommendation(self, score: float) -> str:
        """Get investment recommendation based on score"""
        
        if score >= 80:
            return 'Strong Buy - Excellent investment opportunity'
        elif score >= 60:
            return 'Buy - Good investment potential'
        elif score >= 40:
            return 'Hold - Average investment, monitor market'
        else:
            return 'Avoid - Poor investment metrics'
    
    def _identify_investment_risks(self, property_obj: Property) -> List[str]:
        """Identify potential investment risks"""
        
        risks = []
        
        if property_obj.year_built and property_obj.year_built < 1980:
            risks.append('Older property may require significant maintenance')
        
        if property_obj.occupancy_rate < 90:
            risks.append('Below-average occupancy rate')
        
        if property_obj.days_on_market > 90:
            risks.append('Property has been on market for extended period')
        
        return risks
    
    def _identify_investment_opportunities(self, property_obj: Property, valuation_analysis: Dict) -> List[str]:
        """Identify investment opportunities"""
        
        opportunities = []
        
        if valuation_analysis['value_difference_percent'] < -10:
            opportunities.append('Property appears undervalued by current market standards')
        
        if property_obj.monthly_rent and property_obj.monthly_rent < property_obj.current_value * 0.01:
            opportunities.append('Rent appears below market rate - potential for increase')
        
        if property_obj.property_type == PropertyType.RESIDENTIAL and property_obj.square_footage and property_obj.square_footage > 2500:
            opportunities.append('Large property suitable for subdivision or multi-family conversion')
        
        return opportunities
    
    def optimize_tenant_management(self, agency_id: str) -> Dict[str, Any]:
        """AI-powered tenant management optimization"""
        
        # Get all properties and tenants for the agency
        properties = Property.query.filter_by(agency_id=agency_id).all()
        all_tenants = []
        
        for prop in properties:
            tenants = Tenant.query.filter_by(property_id=prop.property_id).all()
            all_tenants.extend(tenants)
        
        if not all_tenants:
            return {'error': 'No tenants found'}
        
        # Analyze tenant performance
        tenant_analysis = self._analyze_tenant_performance(all_tenants)
        
        # Identify risk factors
        risk_analysis = self._analyze_tenant_risks(all_tenants)
        
        # Generate recommendations
        recommendations = self._generate_tenant_recommendations(tenant_analysis, risk_analysis)
        
        return {
            'agency_id': agency_id,
            'total_tenants': len(all_tenants),
            'tenant_performance_analysis': tenant_analysis,
            'risk_analysis': risk_analysis,
            'optimization_recommendations': recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_tenant_performance(self, tenants: List[Tenant]) -> Dict[str, Any]:
        """Analyze overall tenant performance metrics"""
        
        active_tenants = [t for t in tenants if t.status == TenantStatus.ACTIVE]
        
        if not active_tenants:
            return {'error': 'No active tenants'}
        
        # Calculate performance metrics
        avg_credit_score = np.mean([t.credit_score for t in active_tenants if t.credit_score])
        avg_late_payments = np.mean([t.late_payments for t in active_tenants])
        avg_renewal_probability = np.mean([t.renewal_probability for t in active_tenants])
        
        # Payment performance
        on_time_payments = len([t for t in active_tenants if t.late_payments == 0])
        payment_performance = (on_time_payments / len(active_tenants)) * 100
        
        # Risk distribution
        high_risk_tenants = len([t for t in active_tenants if t.risk_score > 70])
        medium_risk_tenants = len([t for t in active_tenants if 30 <= t.risk_score <= 70])
        low_risk_tenants = len([t for t in active_tenants if t.risk_score < 30])
        
        return {
            'active_tenants': len(active_tenants),
            'average_credit_score': avg_credit_score,
            'average_late_payments': avg_late_payments,
            'payment_performance_percent': payment_performance,
            'average_renewal_probability': avg_renewal_probability,
            'risk_distribution': {
                'high_risk': high_risk_tenants,
                'medium_risk': medium_risk_tenants,
                'low_risk': low_risk_tenants
            }
        }
    
    def _analyze_tenant_risks(self, tenants: List[Tenant]) -> Dict[str, Any]:
        """Analyze tenant risk factors"""
        
        risk_factors = {
            'payment_risks': [],
            'renewal_risks': [],
            'credit_risks': []
        }
        
        for tenant in tenants:
            if tenant.status == TenantStatus.ACTIVE:
                # Payment risk
                if tenant.late_payments > 2:
                    risk_factors['payment_risks'].append({
                        'tenant_id': tenant.tenant_id,
                        'tenant_name': tenant.name,
                        'late_payments': tenant.late_payments,
                        'risk_score': tenant.risk_score
                    })
                
                # Renewal risk
                if tenant.renewal_probability < 50:
                    lease_end = tenant.lease_end_date
                    months_until_end = (lease_end - datetime.now().date()).days / 30 if lease_end else 12
                    
                    risk_factors['renewal_risks'].append({
                        'tenant_id': tenant.tenant_id,
                        'tenant_name': tenant.name,
                        'renewal_probability': tenant.renewal_probability,
                        'months_until_lease_end': months_until_end
                    })
                
                # Credit risk
                if tenant.credit_score and tenant.credit_score < 650:
                    risk_factors['credit_risks'].append({
                        'tenant_id': tenant.tenant_id,
                        'tenant_name': tenant.name,
                        'credit_score': tenant.credit_score,
                        'risk_score': tenant.risk_score
                    })
        
        return risk_factors
    
    def _generate_tenant_recommendations(self, tenant_analysis: Dict, risk_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate tenant management recommendations"""
        
        recommendations = []
        
        # Payment performance recommendations
        if tenant_analysis.get('payment_performance_percent', 100) < 85:
            recommendations.append({
                'category': 'payment_management',
                'priority': 'high',
                'recommendation': 'Implement automated payment reminders and incentives for on-time payments',
                'expected_impact': 'Reduce late payments by 30-50%'
            })
        
        # Risk management recommendations
        high_risk_count = tenant_analysis.get('risk_distribution', {}).get('high_risk', 0)
        if high_risk_count > 0:
            recommendations.append({
                'category': 'risk_management',
                'priority': 'high',
                'recommendation': f'Review {high_risk_count} high-risk tenants for potential intervention',
                'expected_impact': 'Prevent potential defaults and vacancies'
            })
        
        # Renewal optimization
        if len(risk_analysis.get('renewal_risks', [])) > 0:
            recommendations.append({
                'category': 'retention',
                'priority': 'medium',
                'recommendation': 'Proactively engage with at-risk tenants to improve renewal rates',
                'expected_impact': 'Increase renewal rates by 15-25%'
            })
        
        # Credit screening improvements
        if tenant_analysis.get('average_credit_score', 750) < 680:
            recommendations.append({
                'category': 'screening',
                'priority': 'medium',
                'recommendation': 'Strengthen tenant screening criteria and processes',
                'expected_impact': 'Improve tenant quality and reduce risks'
            })
        
        return recommendations

# Initialize engine
real_estate_engine = RealEstateIntelligenceEngine()

# Routes
@app.route('/real-estate')
def real_estate_dashboard():
    """Real Estate & Property Management dashboard"""
    
    recent_agencies = RealEstateAgency.query.order_by(RealEstateAgency.created_at.desc()).limit(10).all()
    
    return render_template('real_estate/dashboard.html',
                         agencies=recent_agencies)

@app.route('/real-estate/api/property-valuation', methods=['POST'])
def analyze_property():
    """API endpoint for property valuation"""
    
    data = request.get_json()
    property_id = data.get('property_id')
    
    if not property_id:
        return jsonify({'error': 'Property ID required'}), 400
    
    analysis = real_estate_engine.analyze_property_valuation(property_id)
    return jsonify(analysis)

@app.route('/real-estate/api/tenant-optimization', methods=['POST'])
def optimize_tenants():
    """API endpoint for tenant management optimization"""
    
    data = request.get_json()
    agency_id = data.get('agency_id')
    
    if not agency_id:
        return jsonify({'error': 'Agency ID required'}), 400
    
    optimization = real_estate_engine.optimize_tenant_management(agency_id)
    return jsonify(optimization)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if RealEstateAgency.query.count() == 0:
        sample_agency = RealEstateAgency(
            agency_id='RE_DEMO_001',
            agency_name='Demo Realty Group',
            location='Metropolitan Area',
            property_count=125,
            agent_count=8,
            specializations=['residential', 'commercial'],
            monthly_revenue=185000,
            average_sale_price=425000
        )
        
        db.session.add(sample_agency)
        db.session.commit()
        logger.info("Sample real estate data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5024, debug=True)