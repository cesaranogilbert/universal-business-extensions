"""
SMB Operations Intelligence Suite - Complete AI Platform for Small-Medium Businesses
$20B+ Value Potential - Comprehensive Operations Automation for 30M+ SMBs
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
app.secret_key = os.environ.get("SESSION_SECRET", "smb-operations-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///smb_operations.db")

db.init_app(app)

# Business Enums
class BusinessSize(Enum):
    MICRO = "micro"  # 1-9 employees
    SMALL = "small"  # 10-49 employees
    MEDIUM = "medium"  # 50-249 employees

class InventoryStatus(Enum):
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    OVERSTOCKED = "overstocked"

# Data Models
class SMBusiness(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(100), unique=True, nullable=False)
    business_name = db.Column(db.String(200), nullable=False)
    industry = db.Column(db.String(100), nullable=False)
    business_size = db.Column(db.Enum(BusinessSize), nullable=False)
    
    # Business Details
    employee_count = db.Column(db.Integer, default=1)
    annual_revenue = db.Column(db.Float, default=0.0)
    location = db.Column(db.String(200))
    established_date = db.Column(db.Date)
    
    # Configuration
    ai_preferences = db.Column(db.JSON)
    subscription_tier = db.Column(db.String(50), default='basic')
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class InventoryItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('sm_business.business_id'), nullable=False)
    
    # Product Details
    name = db.Column(db.String(200), nullable=False)
    sku = db.Column(db.String(100), unique=True)
    category = db.Column(db.String(100))
    description = db.Column(db.Text)
    
    # Inventory Management
    current_stock = db.Column(db.Integer, default=0)
    reorder_point = db.Column(db.Integer, default=10)
    max_stock = db.Column(db.Integer, default=100)
    unit_cost = db.Column(db.Float, default=0.0)
    selling_price = db.Column(db.Float, default=0.0)
    
    # AI Predictions
    predicted_demand = db.Column(db.Integer, default=0)
    optimal_reorder_quantity = db.Column(db.Integer, default=20)
    status = db.Column(db.Enum(InventoryStatus), default=InventoryStatus.IN_STOCK)
    
    # Supplier Information
    supplier_name = db.Column(db.String(200))
    lead_time_days = db.Column(db.Integer, default=7)
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class CustomerProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('sm_business.business_id'), nullable=False)
    
    # Customer Details
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    address = db.Column(db.Text)
    
    # Behavior Analytics
    total_purchases = db.Column(db.Float, default=0.0)
    purchase_frequency = db.Column(db.Float, default=0.0)  # purchases per month
    average_order_value = db.Column(db.Float, default=0.0)
    customer_lifetime_value = db.Column(db.Float, default=0.0)
    
    # AI Insights
    customer_segment = db.Column(db.String(50))  # high_value, regular, occasional
    churn_probability = db.Column(db.Float, default=0.0)
    next_purchase_prediction = db.Column(db.Date)
    recommended_products = db.Column(db.JSON)
    
    # Engagement
    last_interaction = db.Column(db.DateTime)
    preferred_communication = db.Column(db.String(50))  # email, sms, phone
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FinancialMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    metric_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('sm_business.business_id'), nullable=False)
    
    # Financial Data
    date = db.Column(db.Date, nullable=False)
    revenue = db.Column(db.Float, default=0.0)
    expenses = db.Column(db.Float, default=0.0)
    profit = db.Column(db.Float, default=0.0)
    cash_flow = db.Column(db.Float, default=0.0)
    
    # Detailed Breakdown
    expense_breakdown = db.Column(db.JSON)  # categories and amounts
    revenue_sources = db.Column(db.JSON)  # different income streams
    
    # AI Predictions
    predicted_revenue_next_month = db.Column(db.Float, default=0.0)
    predicted_expenses_next_month = db.Column(db.Float, default=0.0)
    cash_flow_forecast = db.Column(db.JSON)  # 3-month forecast
    financial_health_score = db.Column(db.Float, default=75.0)
    
    # Recommendations
    optimization_suggestions = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmployeeMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(100), nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('sm_business.business_id'), nullable=False)
    
    # Employee Details
    name = db.Column(db.String(200), nullable=False)
    position = db.Column(db.String(100))
    department = db.Column(db.String(100))
    hire_date = db.Column(db.Date)
    
    # Productivity Metrics
    daily_tasks_completed = db.Column(db.Integer, default=0)
    quality_score = db.Column(db.Float, default=80.0)  # 0-100
    customer_satisfaction = db.Column(db.Float, default=85.0)  # 0-100
    punctuality_score = db.Column(db.Float, default=95.0)  # 0-100
    
    # AI Analysis
    productivity_trend = db.Column(db.String(50))  # improving, stable, declining
    performance_predictions = db.Column(db.JSON)
    training_recommendations = db.Column(db.JSON)
    
    # Engagement
    satisfaction_level = db.Column(db.Float, default=75.0)
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# SMB Operations Intelligence Engine
class SMBOperationsEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def analyze_inventory_optimization(self, business_id: str) -> Dict[str, Any]:
        """Comprehensive inventory analysis and optimization"""
        
        items = InventoryItem.query.filter_by(business_id=business_id).all()
        
        if not items:
            return {'error': 'No inventory data found'}
        
        optimization_results = []
        total_savings_potential = 0
        
        for item in items:
            analysis = self._analyze_single_item(item)
            optimization_results.append(analysis)
            total_savings_potential += analysis.get('savings_potential', 0)
        
        # Generate AI recommendations
        recommendations = self._generate_inventory_recommendations(optimization_results)
        
        return {
            'business_id': business_id,
            'total_items_analyzed': len(items),
            'total_savings_potential': total_savings_potential,
            'items_needing_attention': len([r for r in optimization_results if r.get('needs_action', False)]),
            'detailed_analysis': optimization_results,
            'ai_recommendations': recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_single_item(self, item: InventoryItem) -> Dict[str, Any]:
        """Analyze individual inventory item"""
        
        # Calculate key metrics
        turnover_rate = self._calculate_turnover_rate(item)
        carrying_cost = self._calculate_carrying_cost(item)
        stockout_risk = self._calculate_stockout_risk(item)
        
        # AI-powered demand prediction
        demand_prediction = self._predict_demand(item)
        
        # Optimization recommendations
        optimal_stock = self._calculate_optimal_stock_level(item, demand_prediction)
        
        savings_potential = max(0, (item.current_stock - optimal_stock) * item.unit_cost * 0.1)
        
        return {
            'item_id': item.item_id,
            'name': item.name,
            'current_stock': item.current_stock,
            'optimal_stock': optimal_stock,
            'turnover_rate': turnover_rate,
            'carrying_cost': carrying_cost,
            'stockout_risk': stockout_risk,
            'predicted_demand': demand_prediction,
            'savings_potential': savings_potential,
            'needs_action': item.current_stock < item.reorder_point or item.current_stock > item.max_stock * 1.2,
            'recommendations': self._generate_item_recommendations(item, optimal_stock)
        }
    
    def _calculate_turnover_rate(self, item: InventoryItem) -> float:
        """Calculate inventory turnover rate"""
        # Simplified calculation - in production, use actual sales data
        annual_demand = item.predicted_demand * 12 if item.predicted_demand > 0 else 50
        average_inventory = (item.current_stock + item.reorder_point) / 2
        return annual_demand / average_inventory if average_inventory > 0 else 0
    
    def _calculate_carrying_cost(self, item: InventoryItem) -> float:
        """Calculate carrying cost for item"""
        # Typically 20-30% of item value annually
        carrying_cost_rate = 0.25
        return item.current_stock * item.unit_cost * carrying_cost_rate
    
    def _calculate_stockout_risk(self, item: InventoryItem) -> float:
        """Calculate probability of stockout"""
        days_of_supply = item.current_stock / max(1, item.predicted_demand / 30)
        if days_of_supply > item.lead_time_days * 2:
            return 0.1  # Low risk
        elif days_of_supply > item.lead_time_days:
            return 0.3  # Medium risk
        else:
            return 0.8  # High risk
    
    def _predict_demand(self, item: InventoryItem) -> int:
        """AI-powered demand prediction"""
        # Simplified prediction - in production, use historical sales data
        base_demand = max(1, item.current_stock // 10)
        seasonal_factor = 1.0  # Could include seasonal adjustments
        return int(base_demand * seasonal_factor)
    
    def _calculate_optimal_stock_level(self, item: InventoryItem, predicted_demand: int) -> int:
        """Calculate optimal stock level using EOQ and safety stock"""
        # Economic Order Quantity calculation
        annual_demand = predicted_demand * 12
        ordering_cost = 50  # Assumed ordering cost
        carrying_cost_per_unit = item.unit_cost * 0.25
        
        if carrying_cost_per_unit > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / carrying_cost_per_unit)
        else:
            eoq = item.reorder_point
        
        # Safety stock
        safety_stock = predicted_demand * item.lead_time_days / 30
        
        return int(eoq + safety_stock)
    
    def _generate_item_recommendations(self, item: InventoryItem, optimal_stock: int) -> List[str]:
        """Generate specific recommendations for item"""
        recommendations = []
        
        if item.current_stock > optimal_stock * 1.5:
            recommendations.append(f"Consider reducing stock by {item.current_stock - optimal_stock} units")
        
        if item.current_stock < item.reorder_point:
            recommendations.append(f"Immediate reorder needed - current stock below reorder point")
        
        if item.predicted_demand == 0:
            recommendations.append("Review demand patterns - item may be obsolete")
        
        return recommendations
    
    def _generate_inventory_recommendations(self, optimization_results: List[Dict[str, Any]]) -> List[str]:
        """Generate overall inventory management recommendations"""
        
        prompt = f"""
        Analyze the following inventory optimization results and provide strategic recommendations:
        
        Inventory Analysis: {json.dumps(optimization_results[:5], indent=2)}  # First 5 items for analysis
        
        Provide specific recommendations for:
        1. Inventory reduction opportunities
        2. Stock level optimization strategies
        3. Supplier relationship improvements
        4. Cost reduction initiatives
        5. Automation opportunities
        
        Format as actionable bullet points.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are an inventory management expert specializing in small business optimization."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('recommendations', [])
            
        except Exception as e:
            logger.error(f"AI recommendation generation failed: {e}")
            return ["Review inventory levels regularly", "Implement automated reordering", "Monitor demand patterns"]
    
    def analyze_customer_behavior(self, business_id: str) -> Dict[str, Any]:
        """Comprehensive customer behavior analysis"""
        
        customers = CustomerProfile.query.filter_by(business_id=business_id).all()
        
        if not customers:
            return {'error': 'No customer data found'}
        
        # Customer segmentation
        segments = self._segment_customers(customers)
        
        # Churn analysis
        churn_analysis = self._analyze_churn_risk(customers)
        
        # Revenue optimization
        revenue_opportunities = self._identify_revenue_opportunities(customers)
        
        # AI insights
        ai_insights = self._generate_customer_insights(customers)
        
        return {
            'business_id': business_id,
            'total_customers': len(customers),
            'customer_segments': segments,
            'churn_analysis': churn_analysis,
            'revenue_opportunities': revenue_opportunities,
            'ai_insights': ai_insights,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _segment_customers(self, customers: List[CustomerProfile]) -> Dict[str, Any]:
        """Segment customers based on value and behavior"""
        
        high_value = [c for c in customers if c.customer_lifetime_value > 1000]
        regular = [c for c in customers if 200 <= c.customer_lifetime_value <= 1000]
        occasional = [c for c in customers if c.customer_lifetime_value < 200]
        
        return {
            'high_value': {
                'count': len(high_value),
                'total_value': sum(c.customer_lifetime_value for c in high_value),
                'avg_order_value': np.mean([c.average_order_value for c in high_value]) if high_value else 0
            },
            'regular': {
                'count': len(regular),
                'total_value': sum(c.customer_lifetime_value for c in regular),
                'avg_order_value': np.mean([c.average_order_value for c in regular]) if regular else 0
            },
            'occasional': {
                'count': len(occasional),
                'total_value': sum(c.customer_lifetime_value for c in occasional),
                'avg_order_value': np.mean([c.average_order_value for c in occasional]) if occasional else 0
            }
        }
    
    def _analyze_churn_risk(self, customers: List[CustomerProfile]) -> Dict[str, Any]:
        """Analyze customer churn risk"""
        
        high_risk = [c for c in customers if c.churn_probability > 0.7]
        medium_risk = [c for c in customers if 0.3 <= c.churn_probability <= 0.7]
        low_risk = [c for c in customers if c.churn_probability < 0.3]
        
        return {
            'high_risk_customers': len(high_risk),
            'medium_risk_customers': len(medium_risk),
            'low_risk_customers': len(low_risk),
            'revenue_at_risk': sum(c.customer_lifetime_value for c in high_risk),
            'retention_strategies_needed': len(high_risk) + len(medium_risk)
        }
    
    def _identify_revenue_opportunities(self, customers: List[CustomerProfile]) -> Dict[str, Any]:
        """Identify revenue growth opportunities"""
        
        upsell_candidates = [c for c in customers if c.average_order_value < 100 and c.purchase_frequency > 2]
        cross_sell_candidates = [c for c in customers if len(c.recommended_products or []) > 0]
        reactivation_candidates = [c for c in customers if c.last_interaction and 
                                 (datetime.utcnow() - c.last_interaction).days > 90]
        
        return {
            'upsell_opportunities': len(upsell_candidates),
            'cross_sell_opportunities': len(cross_sell_candidates),
            'reactivation_opportunities': len(reactivation_candidates),
            'potential_additional_revenue': len(upsell_candidates) * 50 + len(cross_sell_candidates) * 30
        }
    
    def _generate_customer_insights(self, customers: List[CustomerProfile]) -> List[str]:
        """Generate AI-powered customer insights"""
        
        customer_summary = {
            'total_customers': len(customers),
            'avg_clv': np.mean([c.customer_lifetime_value for c in customers]),
            'avg_order_value': np.mean([c.average_order_value for c in customers]),
            'avg_frequency': np.mean([c.purchase_frequency for c in customers])
        }
        
        prompt = f"""
        Analyze customer data and provide insights:
        
        Customer Summary: {json.dumps(customer_summary, indent=2)}
        
        Provide insights on:
        1. Customer retention strategies
        2. Revenue growth opportunities
        3. Marketing campaign recommendations
        4. Customer service improvements
        5. Pricing optimization suggestions
        
        Format as actionable insights.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a customer analytics expert specializing in small business growth."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('insights', [])
            
        except Exception as e:
            logger.error(f"Customer insights generation failed: {e}")
            return ["Focus on customer retention", "Implement loyalty programs", "Personalize customer communications"]
    
    def analyze_financial_health(self, business_id: str) -> Dict[str, Any]:
        """Comprehensive financial health analysis"""
        
        # Get recent financial data
        recent_metrics = FinancialMetric.query.filter_by(business_id=business_id)\
                                               .order_by(FinancialMetric.date.desc())\
                                               .limit(12).all()
        
        if not recent_metrics:
            return {'error': 'No financial data found'}
        
        # Calculate key financial ratios
        financial_ratios = self._calculate_financial_ratios(recent_metrics)
        
        # Cash flow analysis
        cash_flow_analysis = self._analyze_cash_flow(recent_metrics)
        
        # Trend analysis
        trend_analysis = self._analyze_financial_trends(recent_metrics)
        
        # AI-powered recommendations
        recommendations = self._generate_financial_recommendations(recent_metrics, financial_ratios)
        
        return {
            'business_id': business_id,
            'analysis_period': f"{recent_metrics[-1].date} to {recent_metrics[0].date}",
            'financial_health_score': recent_metrics[0].financial_health_score,
            'financial_ratios': financial_ratios,
            'cash_flow_analysis': cash_flow_analysis,
            'trend_analysis': trend_analysis,
            'recommendations': recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _calculate_financial_ratios(self, metrics: List[FinancialMetric]) -> Dict[str, float]:
        """Calculate key financial ratios"""
        
        latest = metrics[0]
        
        # Profit margin
        profit_margin = (latest.profit / latest.revenue * 100) if latest.revenue > 0 else 0
        
        # Revenue growth (comparing to previous month)
        revenue_growth = 0
        if len(metrics) > 1:
            previous_revenue = metrics[1].revenue
            if previous_revenue > 0:
                revenue_growth = ((latest.revenue - previous_revenue) / previous_revenue * 100)
        
        # Operating efficiency
        operating_efficiency = (latest.revenue / latest.expenses) if latest.expenses > 0 else 0
        
        return {
            'profit_margin_percent': round(profit_margin, 2),
            'revenue_growth_percent': round(revenue_growth, 2),
            'operating_efficiency_ratio': round(operating_efficiency, 2),
            'monthly_burn_rate': latest.expenses
        }
    
    def _analyze_cash_flow(self, metrics: List[FinancialMetric]) -> Dict[str, Any]:
        """Analyze cash flow patterns"""
        
        cash_flows = [m.cash_flow for m in metrics]
        
        return {
            'current_cash_flow': cash_flows[0],
            'average_monthly_cash_flow': np.mean(cash_flows),
            'cash_flow_volatility': np.std(cash_flows),
            'positive_cash_flow_months': len([cf for cf in cash_flows if cf > 0]),
            'cash_runway_months': self._calculate_cash_runway(metrics)
        }
    
    def _calculate_cash_runway(self, metrics: List[FinancialMetric]) -> int:
        """Calculate how many months the business can survive"""
        
        avg_monthly_burn = np.mean([m.expenses - m.revenue for m in metrics if m.expenses > m.revenue])
        
        if avg_monthly_burn <= 0:
            return 999  # Profitable or break-even
        
        # Assume current cash is 3x monthly revenue (simplified)
        estimated_cash = metrics[0].revenue * 3
        
        return int(estimated_cash / avg_monthly_burn) if avg_monthly_burn > 0 else 999
    
    def _analyze_financial_trends(self, metrics: List[FinancialMetric]) -> Dict[str, str]:
        """Analyze financial trends"""
        
        revenues = [m.revenue for m in reversed(metrics)]
        expenses = [m.expenses for m in reversed(metrics)]
        profits = [m.profit for m in reversed(metrics)]
        
        return {
            'revenue_trend': self._calculate_trend(revenues),
            'expense_trend': self._calculate_trend(expenses),
            'profit_trend': self._calculate_trend(profits)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = np.mean(values[:-3]) if len(values) >= 6 else values[0]
        
        change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        if change_percent > 10:
            return 'strongly_increasing'
        elif change_percent > 2:
            return 'increasing'
        elif change_percent < -10:
            return 'strongly_decreasing'
        elif change_percent < -2:
            return 'decreasing'
        else:
            return 'stable'
    
    def _generate_financial_recommendations(self, metrics: List[FinancialMetric], ratios: Dict[str, float]) -> List[str]:
        """Generate AI-powered financial recommendations"""
        
        prompt = f"""
        Analyze financial performance and provide recommendations:
        
        Financial Ratios: {json.dumps(ratios, indent=2)}
        Recent Performance: Revenue: ${metrics[0].revenue}, Expenses: ${metrics[0].expenses}, Profit: ${metrics[0].profit}
        
        Provide recommendations for:
        1. Cost reduction opportunities
        2. Revenue growth strategies
        3. Cash flow optimization
        4. Financial risk mitigation
        5. Investment priorities
        
        Format as specific, actionable recommendations.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a financial advisor specializing in small business optimization."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('recommendations', [])
            
        except Exception as e:
            logger.error(f"Financial recommendations failed: {e}")
            return ["Monitor cash flow closely", "Reduce unnecessary expenses", "Focus on customer retention"]

# Initialize engine
operations_engine = SMBOperationsEngine()

# Routes
@app.route('/smb-operations')
def operations_dashboard():
    """SMB Operations Intelligence dashboard"""
    
    # Get sample business data
    recent_businesses = SMBusiness.query.order_by(SMBusiness.created_at.desc()).limit(10).all()
    
    return render_template('smb_operations/dashboard.html',
                         businesses=recent_businesses)

@app.route('/smb-operations/api/inventory-analysis', methods=['POST'])
def analyze_inventory():
    """API endpoint for inventory analysis"""
    
    data = request.get_json()
    business_id = data.get('business_id')
    
    if not business_id:
        return jsonify({'error': 'Business ID required'}), 400
    
    analysis = operations_engine.analyze_inventory_optimization(business_id)
    return jsonify(analysis)

@app.route('/smb-operations/api/customer-analysis', methods=['POST'])
def analyze_customers():
    """API endpoint for customer analysis"""
    
    data = request.get_json()
    business_id = data.get('business_id')
    
    if not business_id:
        return jsonify({'error': 'Business ID required'}), 400
    
    analysis = operations_engine.analyze_customer_behavior(business_id)
    return jsonify(analysis)

@app.route('/smb-operations/api/financial-analysis', methods=['POST'])
def analyze_finances():
    """API endpoint for financial analysis"""
    
    data = request.get_json()
    business_id = data.get('business_id')
    
    if not business_id:
        return jsonify({'error': 'Business ID required'}), 400
    
    analysis = operations_engine.analyze_financial_health(business_id)
    return jsonify(analysis)

@app.route('/smb-operations/api/register-business', methods=['POST'])
def register_business():
    """Register new SMB for operations intelligence"""
    
    data = request.get_json()
    
    business = SMBusiness(
        business_id=f"SMB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        business_name=data.get('business_name'),
        industry=data.get('industry'),
        business_size=BusinessSize(data.get('business_size', 'small')),
        employee_count=data.get('employee_count', 1),
        annual_revenue=data.get('annual_revenue', 0),
        location=data.get('location')
    )
    
    db.session.add(business)
    db.session.commit()
    
    return jsonify({
        'business_id': business.business_id,
        'status': 'registered',
        'message': 'SMB registered successfully for operations intelligence'
    })

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if SMBusiness.query.count() == 0:
        sample_business = SMBusiness(
            business_id='SMB_DEMO_001',
            business_name='Demo Restaurant',
            industry='Food Service',
            business_size=BusinessSize.SMALL,
            employee_count=15,
            annual_revenue=500000,
            location='Downtown'
        )
        
        db.session.add(sample_business)
        
        # Sample inventory items
        inventory_items = [
            InventoryItem(
                item_id='ITEM_001',
                business_id='SMB_DEMO_001',
                name='Tomatoes',
                sku='TOM-001',
                category='Produce',
                current_stock=50,
                reorder_point=20,
                unit_cost=2.50,
                selling_price=4.00,
                supplier_name='Fresh Produce Co'
            ),
            InventoryItem(
                item_id='ITEM_002',
                business_id='SMB_DEMO_001',
                name='Chicken Breast',
                sku='CHK-001',
                category='Meat',
                current_stock=30,
                reorder_point=15,
                unit_cost=8.00,
                selling_price=15.00,
                supplier_name='Quality Meats Inc'
            )
        ]
        
        for item in inventory_items:
            db.session.add(item)
        
        db.session.commit()
        logger.info("Sample SMB operations data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5020, debug=True)