"""
Food Service & Hospitality Optimization Platform - Complete AI-Powered Restaurant & Hotel System
$18B+ Value Potential - Menu Engineering, Staff Optimization & Customer Experience Management
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
app.secret_key = os.environ.get("SESSION_SECRET", "hospitality-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///hospitality.db")

db.init_app(app)

# Hospitality Enums
class BusinessType(Enum):
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BAR = "bar"
    HOTEL = "hotel"
    CATERING = "catering"
    FOOD_TRUCK = "food_truck"

class OrderStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    SERVED = "served"
    CANCELLED = "cancelled"

class ReservationStatus(Enum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    SEATED = "seated"

# Data Models
class HospitalityBusiness(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_id = db.Column(db.String(100), unique=True, nullable=False)
    business_name = db.Column(db.String(200), nullable=False)
    business_type = db.Column(db.Enum(BusinessType), nullable=False)
    
    # Location and Capacity
    address = db.Column(db.String(500))
    seating_capacity = db.Column(db.Integer, default=50)
    room_count = db.Column(db.Integer, default=0)  # For hotels
    
    # Operating Hours
    opening_hours = db.Column(db.JSON)  # Day-by-day schedule
    
    # Performance Metrics
    average_ticket_size = db.Column(db.Float, default=25.0)
    daily_covers = db.Column(db.Integer, default=100)  # customers served per day
    table_turnover_rate = db.Column(db.Float, default=2.5)  # times per day
    customer_satisfaction = db.Column(db.Float, default=4.2)  # 1-5 scale
    
    # Financial Metrics
    daily_revenue = db.Column(db.Float, default=2500.0)
    food_cost_percentage = db.Column(db.Float, default=30.0)
    labor_cost_percentage = db.Column(db.Float, default=35.0)
    profit_margin = db.Column(db.Float, default=15.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MenuItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('hospitality_business.business_id'), nullable=False)
    
    # Item Details
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(100))  # appetizer, entree, dessert, beverage, etc.
    
    # Pricing
    menu_price = db.Column(db.Float, nullable=False)
    cost_to_make = db.Column(db.Float, default=0.0)
    profit_margin = db.Column(db.Float, default=0.0)
    
    # Popularity and Performance
    daily_sales_count = db.Column(db.Integer, default=0)
    weekly_sales_count = db.Column(db.Integer, default=0)
    customer_rating = db.Column(db.Float, default=4.0)  # 1-5 scale
    
    # Preparation Details
    prep_time_minutes = db.Column(db.Integer, default=15)
    skill_level_required = db.Column(db.String(50), default='medium')  # easy, medium, hard
    ingredients = db.Column(db.JSON)  # List of ingredients with quantities
    
    # Menu Engineering
    menu_mix_percentage = db.Column(db.Float, default=0.0)  # % of total sales
    popularity_rank = db.Column(db.Integer, default=50)
    profitability_rank = db.Column(db.Integer, default=50)
    menu_category_ranking = db.Column(db.String(50))  # star, plow_horse, puzzle, dog
    
    # AI Recommendations
    recommended_price = db.Column(db.Float, default=0.0)
    demand_forecast = db.Column(db.JSON)  # 7-day demand prediction
    optimization_suggestions = db.Column(db.JSON)
    
    is_available = db.Column(db.Boolean, default=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('hospitality_business.business_id'), nullable=False)
    
    # Order Details
    order_time = db.Column(db.DateTime, default=datetime.utcnow)
    table_number = db.Column(db.String(20))
    party_size = db.Column(db.Integer, default=1)
    
    # Items Ordered
    items = db.Column(db.JSON)  # List of item_id, quantity, special_instructions
    
    # Financials
    subtotal = db.Column(db.Float, default=0.0)
    tax_amount = db.Column(db.Float, default=0.0)
    tip_amount = db.Column(db.Float, default=0.0)
    total_amount = db.Column(db.Float, default=0.0)
    
    # Service Details
    server_id = db.Column(db.String(100))
    kitchen_staff_assigned = db.Column(db.JSON)
    
    # Timing
    order_placed_time = db.Column(db.DateTime)
    kitchen_start_time = db.Column(db.DateTime)
    ready_time = db.Column(db.DateTime)
    served_time = db.Column(db.DateTime)
    
    # Status and Quality
    status = db.Column(db.Enum(OrderStatus), default=OrderStatus.PENDING)
    customer_satisfaction = db.Column(db.Float, default=0.0)  # 1-5 scale
    special_requests = db.Column(db.Text)
    
    # AI Analysis
    wait_time_minutes = db.Column(db.Float, default=0.0)
    service_quality_score = db.Column(db.Float, default=85.0)  # 0-100

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('hospitality_business.business_id'), nullable=False)
    
    # Customer Information
    name = db.Column(db.String(200))
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    
    # Visit History
    first_visit_date = db.Column(db.Date)
    last_visit_date = db.Column(db.Date)
    total_visits = db.Column(db.Integer, default=1)
    total_spent = db.Column(db.Float, default=0.0)
    
    # Preferences
    favorite_items = db.Column(db.JSON)  # Most ordered items
    dietary_restrictions = db.Column(db.JSON)
    preferred_seating = db.Column(db.String(100))
    usual_party_size = db.Column(db.Integer, default=2)
    
    # Loyalty Metrics
    lifetime_value = db.Column(db.Float, default=0.0)
    loyalty_tier = db.Column(db.String(50), default='regular')  # new, regular, vip
    referrals_made = db.Column(db.Integer, default=0)
    
    # AI Insights
    churn_probability = db.Column(db.Float, default=0.25)  # 0-1
    next_visit_prediction = db.Column(db.Date)
    recommended_offers = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Staff(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('hospitality_business.business_id'), nullable=False)
    
    # Personal Information
    name = db.Column(db.String(200), nullable=False)
    position = db.Column(db.String(100))  # server, cook, manager, host, etc.
    hire_date = db.Column(db.Date)
    
    # Scheduling
    weekly_hours_scheduled = db.Column(db.Float, default=40.0)
    hourly_rate = db.Column(db.Float, default=15.0)
    availability = db.Column(db.JSON)  # Days and hours available
    
    # Performance Metrics
    efficiency_score = db.Column(db.Float, default=85.0)  # 0-100
    customer_rating = db.Column(db.Float, default=4.0)  # 1-5 scale
    attendance_rate = db.Column(db.Float, default=95.0)  # percentage
    
    # Service Metrics (for servers)
    average_table_turnover = db.Column(db.Float, default=2.0)  # times per shift
    average_ticket_size = db.Column(db.Float, default=25.0)
    tips_earned_daily = db.Column(db.Float, default=80.0)
    
    # Kitchen Metrics (for cooks)
    dishes_per_hour = db.Column(db.Float, default=15.0)
    order_accuracy = db.Column(db.Float, default=95.0)  # percentage
    food_waste_percentage = db.Column(db.Float, default=5.0)
    
    # AI Analysis
    performance_trend = db.Column(db.String(50), default='stable')  # improving, stable, declining
    optimal_shifts = db.Column(db.JSON)  # Best performing shift times
    training_recommendations = db.Column(db.JSON)
    
    is_active = db.Column(db.Boolean, default=True)

class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    inventory_id = db.Column(db.String(100), unique=True, nullable=False)
    business_id = db.Column(db.String(100), db.ForeignKey('hospitality_business.business_id'), nullable=False)
    
    # Item Details
    ingredient_name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100))  # protein, vegetable, dairy, etc.
    unit_of_measure = db.Column(db.String(50))  # pounds, ounces, pieces, etc.
    
    # Inventory Levels
    current_quantity = db.Column(db.Float, default=0.0)
    reorder_point = db.Column(db.Float, default=10.0)
    maximum_stock = db.Column(db.Float, default=100.0)
    
    # Cost Information
    unit_cost = db.Column(db.Float, default=1.0)
    supplier_name = db.Column(db.String(200))
    last_order_date = db.Column(db.Date)
    
    # Usage Patterns
    daily_usage_average = db.Column(db.Float, default=5.0)
    weekly_usage_trend = db.Column(db.String(50), default='stable')
    
    # Quality and Shelf Life
    expiration_date = db.Column(db.Date)
    quality_rating = db.Column(db.Float, default=9.0)  # 1-10 scale
    storage_requirements = db.Column(db.String(200))
    
    # AI Predictions
    predicted_usage_7_days = db.Column(db.Float, default=35.0)
    optimal_order_quantity = db.Column(db.Float, default=50.0)
    waste_reduction_suggestions = db.Column(db.JSON)
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# Hospitality Intelligence Engine
class HospitalityIntelligenceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def optimize_menu_engineering(self, business_id: str) -> Dict[str, Any]:
        """AI-powered menu engineering and profitability optimization"""
        
        menu_items = MenuItem.query.filter_by(business_id=business_id).all()
        
        if not menu_items:
            return {'error': 'No menu items found'}
        
        # Calculate menu engineering metrics for each item
        menu_analysis = []
        total_weekly_sales = sum(item.weekly_sales_count for item in menu_items)
        
        for item in menu_items:
            analysis = self._analyze_menu_item_performance(item, total_weekly_sales)
            menu_analysis.append(analysis)
        
        # Categorize items using menu engineering matrix
        categorized_items = self._categorize_menu_items(menu_analysis)
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_menu_optimization_recommendations(categorized_items)
        
        # Calculate potential revenue impact
        revenue_impact = self._calculate_menu_revenue_impact(optimization_recommendations, menu_analysis)
        
        return {
            'business_id': business_id,
            'menu_items_analyzed': len(menu_items),
            'menu_analysis': menu_analysis,
            'categorized_items': categorized_items,
            'optimization_recommendations': optimization_recommendations,
            'revenue_impact': revenue_impact,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_menu_item_performance(self, item: MenuItem, total_sales: int) -> Dict[str, Any]:
        """Analyze individual menu item performance"""
        
        # Calculate popularity (menu mix percentage)
        menu_mix = (item.weekly_sales_count / total_sales * 100) if total_sales > 0 else 0
        
        # Calculate profitability
        profit_per_item = item.menu_price - item.cost_to_make
        profit_margin_percent = (profit_per_item / item.menu_price * 100) if item.menu_price > 0 else 0
        
        # Weekly revenue contribution
        weekly_revenue = item.weekly_sales_count * item.menu_price
        weekly_profit = item.weekly_sales_count * profit_per_item
        
        return {
            'item_id': item.item_id,
            'item_name': item.name,
            'category': item.category,
            'menu_price': item.menu_price,
            'cost_to_make': item.cost_to_make,
            'profit_per_item': profit_per_item,
            'profit_margin_percent': profit_margin_percent,
            'weekly_sales_count': item.weekly_sales_count,
            'menu_mix_percent': menu_mix,
            'weekly_revenue': weekly_revenue,
            'weekly_profit': weekly_profit,
            'customer_rating': item.customer_rating,
            'prep_time_minutes': item.prep_time_minutes
        }
    
    def _categorize_menu_items(self, menu_analysis: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize menu items using menu engineering matrix"""
        
        # Calculate median values for popularity and profitability
        popularities = [item['menu_mix_percent'] for item in menu_analysis]
        profit_margins = [item['profit_margin_percent'] for item in menu_analysis]
        
        median_popularity = np.median(popularities) if popularities else 0
        median_profitability = np.median(profit_margins) if profit_margins else 0
        
        categories = {
            'stars': [],      # High popularity, High profitability
            'plow_horses': [],  # High popularity, Low profitability
            'puzzles': [],    # Low popularity, High profitability
            'dogs': []        # Low popularity, Low profitability
        }
        
        for item in menu_analysis:
            popularity = item['menu_mix_percent']
            profitability = item['profit_margin_percent']
            
            if popularity >= median_popularity and profitability >= median_profitability:
                categories['stars'].append(item)
            elif popularity >= median_popularity and profitability < median_profitability:
                categories['plow_horses'].append(item)
            elif popularity < median_popularity and profitability >= median_profitability:
                categories['puzzles'].append(item)
            else:
                categories['dogs'].append(item)
        
        return categories
    
    def _generate_menu_optimization_recommendations(self, categorized_items: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate specific menu optimization recommendations"""
        
        recommendations = []
        
        # Recommendations for Stars (promote and maintain)
        for item in categorized_items['stars']:
            recommendations.append({
                'item_name': item['item_name'],
                'category': 'star',
                'recommendation': 'Promote prominently on menu and maintain quality',
                'action_type': 'promote',
                'expected_impact': 'Maintain high revenue contribution',
                'priority': 'low'  # Already performing well
            })
        
        # Recommendations for Plow Horses (increase profitability)
        for item in categorized_items['plow_horses']:
            recommendations.append({
                'item_name': item['item_name'],
                'category': 'plow_horse',
                'recommendation': f'Increase price by 10-15% or reduce costs',
                'action_type': 'price_increase',
                'expected_impact': f'Potential profit increase of ${item["weekly_sales_count"] * 2:.2f} per week',
                'priority': 'high'
            })
        
        # Recommendations for Puzzles (increase popularity)
        for item in categorized_items['puzzles']:
            recommendations.append({
                'item_name': item['item_name'],
                'category': 'puzzle',
                'recommendation': 'Reposition on menu, improve description, or offer promotions',
                'action_type': 'promotion',
                'expected_impact': 'Increase sales volume and total profit',
                'priority': 'medium'
            })
        
        # Recommendations for Dogs (remove or redesign)
        for item in categorized_items['dogs']:
            if item['customer_rating'] < 3.5:
                recommendations.append({
                    'item_name': item['item_name'],
                    'category': 'dog',
                    'recommendation': 'Consider removing from menu',
                    'action_type': 'remove',
                    'expected_impact': 'Reduce menu complexity and focus kitchen resources',
                    'priority': 'medium'
                })
            else:
                recommendations.append({
                    'item_name': item['item_name'],
                    'category': 'dog',
                    'recommendation': 'Redesign recipe to improve profitability and appeal',
                    'action_type': 'redesign',
                    'expected_impact': 'Transform into profitable item',
                    'priority': 'high'
                })
        
        return recommendations
    
    def _calculate_menu_revenue_impact(self, recommendations: List[Dict], menu_analysis: List[Dict]) -> Dict[str, Any]:
        """Calculate potential revenue impact of menu optimizations"""
        
        current_weekly_revenue = sum(item['weekly_revenue'] for item in menu_analysis)
        current_weekly_profit = sum(item['weekly_profit'] for item in menu_analysis)
        
        # Estimate impact of price increases on plow horses
        price_increase_impact = 0
        for rec in recommendations:
            if rec['action_type'] == 'price_increase':
                item_data = next((item for item in menu_analysis if item['item_name'] == rec['item_name']), None)
                if item_data:
                    # Assume 12% price increase with 5% volume reduction
                    new_price = item_data['menu_price'] * 1.12
                    new_volume = item_data['weekly_sales_count'] * 0.95
                    additional_profit = (new_price - item_data['cost_to_make']) * new_volume - item_data['weekly_profit']
                    price_increase_impact += additional_profit
        
        # Estimate impact of promotions on puzzles
        promotion_impact = 0
        for rec in recommendations:
            if rec['action_type'] == 'promotion':
                item_data = next((item for item in menu_analysis if item['item_name'] == rec['item_name']), None)
                if item_data:
                    # Assume 30% volume increase from better positioning
                    additional_volume = item_data['weekly_sales_count'] * 0.30
                    additional_profit = additional_volume * item_data['profit_per_item']
                    promotion_impact += additional_profit
        
        total_estimated_weekly_impact = price_increase_impact + promotion_impact
        annual_impact = total_estimated_weekly_impact * 52
        
        return {
            'current_weekly_revenue': current_weekly_revenue,
            'current_weekly_profit': current_weekly_profit,
            'estimated_weekly_profit_increase': total_estimated_weekly_impact,
            'estimated_annual_profit_increase': annual_impact,
            'roi_improvement_percent': (total_estimated_weekly_impact / current_weekly_profit * 100) if current_weekly_profit > 0 else 0
        }
    
    def optimize_staff_scheduling(self, business_id: str) -> Dict[str, Any]:
        """AI-powered staff scheduling optimization"""
        
        staff_members = Staff.query.filter_by(business_id=business_id, is_active=True).all()
        business = HospitalityBusiness.query.filter_by(business_id=business_id).first()
        
        if not staff_members or not business:
            return {'error': 'Staff or business data not found'}
        
        # Analyze current staffing patterns
        staffing_analysis = self._analyze_current_staffing(staff_members, business)
        
        # Predict demand patterns
        demand_forecast = self._predict_customer_demand(business_id)
        
        # Generate optimal schedule
        optimal_schedule = self._generate_optimal_schedule(staff_members, demand_forecast, business)
        
        # Calculate cost and efficiency improvements
        improvements = self._calculate_scheduling_improvements(staffing_analysis, optimal_schedule)
        
        return {
            'business_id': business_id,
            'staff_count': len(staff_members),
            'current_staffing_analysis': staffing_analysis,
            'demand_forecast': demand_forecast,
            'optimal_schedule': optimal_schedule,
            'improvements': improvements,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_current_staffing(self, staff: List[Staff], business: HospitalityBusiness) -> Dict[str, Any]:
        """Analyze current staffing patterns and efficiency"""
        
        # Calculate staffing costs
        total_weekly_hours = sum(s.weekly_hours_scheduled for s in staff)
        total_weekly_cost = sum(s.weekly_hours_scheduled * s.hourly_rate for s in staff)
        
        # Analyze by position
        positions = {}
        for staff_member in staff:
            position = staff_member.position
            if position not in positions:
                positions[position] = {
                    'count': 0,
                    'total_hours': 0,
                    'total_cost': 0,
                    'avg_efficiency': 0,
                    'avg_rating': 0
                }
            
            positions[position]['count'] += 1
            positions[position]['total_hours'] += staff_member.weekly_hours_scheduled
            positions[position]['total_cost'] += staff_member.weekly_hours_scheduled * staff_member.hourly_rate
            positions[position]['avg_efficiency'] += staff_member.efficiency_score
            positions[position]['avg_rating'] += staff_member.customer_rating
        
        # Calculate averages
        for position_data in positions.values():
            if position_data['count'] > 0:
                position_data['avg_efficiency'] /= position_data['count']
                position_data['avg_rating'] /= position_data['count']
        
        # Calculate labor cost percentage
        labor_cost_percentage = (total_weekly_cost * 52) / (business.daily_revenue * 365) * 100
        
        return {
            'total_weekly_hours': total_weekly_hours,
            'total_weekly_cost': total_weekly_cost,
            'labor_cost_percentage': labor_cost_percentage,
            'staffing_by_position': positions,
            'average_efficiency': np.mean([s.efficiency_score for s in staff]),
            'average_customer_rating': np.mean([s.customer_rating for s in staff])
        }
    
    def _predict_customer_demand(self, business_id: str) -> Dict[str, Any]:
        """Predict customer demand patterns for optimization"""
        
        # Get recent order data
        recent_orders = Order.query.filter_by(business_id=business_id)\
                                  .filter(Order.order_time >= datetime.utcnow() - timedelta(days=30))\
                                  .all()
        
        if not recent_orders:
            # Return default pattern if no data
            return {
                'hourly_demand': {str(i): 50 for i in range(24)},  # Default even distribution
                'daily_demand': {
                    'monday': 80, 'tuesday': 85, 'wednesday': 90,
                    'thursday': 95, 'friday': 120, 'saturday': 130, 'sunday': 100
                }
            }
        
        # Analyze order patterns by hour and day
        hourly_counts = {}
        daily_counts = {}
        
        for order in recent_orders:
            hour = order.order_time.hour
            day = order.order_time.strftime('%A').lower()
            
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            daily_counts[day] = daily_counts.get(day, 0) + 1
        
        # Normalize to percentages
        total_orders = len(recent_orders)
        hourly_demand = {str(hour): (count / total_orders * 100) for hour, count in hourly_counts.items()}
        daily_demand = {day: (count / total_orders * 100) for day, count in daily_counts.items()}
        
        return {
            'hourly_demand': hourly_demand,
            'daily_demand': daily_demand,
            'peak_hours': sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'peak_days': sorted(daily_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _generate_optimal_schedule(self, staff: List[Staff], demand_forecast: Dict, business: HospitalityBusiness) -> Dict[str, Any]:
        """Generate optimal staff schedule based on demand"""
        
        optimal_schedule = {}
        
        # Define shifts
        shifts = {
            'morning': {'start': 6, 'end': 14},
            'afternoon': {'start': 14, 'end': 22},
            'night': {'start': 22, 'end': 6}
        }
        
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        for day in days_of_week:
            day_demand = demand_forecast['daily_demand'].get(day, 100)
            daily_schedule = {}
            
            for shift_name, shift_times in shifts.items():
                # Calculate required staff based on demand
                base_staff_needed = max(2, int(day_demand / 50))  # Minimum 2 staff per shift
                
                # Assign staff based on availability and performance
                assigned_staff = self._assign_staff_to_shift(staff, day, shift_times, base_staff_needed)
                
                daily_schedule[shift_name] = {
                    'required_staff': base_staff_needed,
                    'assigned_staff': assigned_staff,
                    'shift_hours': shift_times
                }
            
            optimal_schedule[day] = daily_schedule
        
        return optimal_schedule
    
    def _assign_staff_to_shift(self, staff: List[Staff], day: str, shift_times: Dict, needed_count: int) -> List[Dict[str, Any]]:
        """Assign staff to specific shift based on availability and performance"""
        
        available_staff = []
        
        for staff_member in staff:
            # Check availability (simplified - assume all staff available for demo)
            availability = staff_member.availability or {}
            if day in availability or not availability:  # If no availability data, assume available
                available_staff.append(staff_member)
        
        # Sort by performance metrics
        available_staff.sort(key=lambda s: (s.efficiency_score + s.customer_rating * 20), reverse=True)
        
        # Assign top performers to shift
        assigned = []
        for i, staff_member in enumerate(available_staff[:needed_count]):
            assigned.append({
                'staff_id': staff_member.staff_id,
                'name': staff_member.name,
                'position': staff_member.position,
                'efficiency_score': staff_member.efficiency_score,
                'hourly_rate': staff_member.hourly_rate
            })
        
        return assigned
    
    def _calculate_scheduling_improvements(self, current_analysis: Dict, optimal_schedule: Dict) -> Dict[str, Any]:
        """Calculate improvements from optimal scheduling"""
        
        # Estimate optimal weekly cost
        optimal_weekly_cost = 0
        optimal_weekly_hours = 0
        
        for day_schedule in optimal_schedule.values():
            for shift_data in day_schedule.values():
                for staff_assignment in shift_data['assigned_staff']:
                    shift_duration = 8  # Assume 8-hour shifts
                    optimal_weekly_hours += shift_duration
                    optimal_weekly_cost += shift_duration * staff_assignment['hourly_rate']
        
        current_weekly_cost = current_analysis['total_weekly_cost']
        current_weekly_hours = current_analysis['total_weekly_hours']
        
        # Calculate savings
        cost_savings = current_weekly_cost - optimal_weekly_cost
        hour_reduction = current_weekly_hours - optimal_weekly_hours
        
        return {
            'current_weekly_cost': current_weekly_cost,
            'optimal_weekly_cost': optimal_weekly_cost,
            'weekly_cost_savings': cost_savings,
            'weekly_hour_reduction': hour_reduction,
            'annual_cost_savings': cost_savings * 52,
            'efficiency_improvement_percent': (cost_savings / current_weekly_cost * 100) if current_weekly_cost > 0 else 0,
            'optimization_benefits': [
                'Better staff-to-demand alignment',
                'Reduced labor costs during slow periods',
                'Improved customer service during peak times',
                'Higher staff utilization efficiency'
            ]
        }

# Initialize engine
hospitality_engine = HospitalityIntelligenceEngine()

# Routes
@app.route('/hospitality')
def hospitality_dashboard():
    """Food Service & Hospitality dashboard"""
    
    recent_businesses = HospitalityBusiness.query.order_by(HospitalityBusiness.created_at.desc()).limit(10).all()
    
    return render_template('hospitality/dashboard.html',
                         businesses=recent_businesses)

@app.route('/hospitality/api/menu-optimization', methods=['POST'])
def optimize_menu():
    """API endpoint for menu engineering optimization"""
    
    data = request.get_json()
    business_id = data.get('business_id')
    
    if not business_id:
        return jsonify({'error': 'Business ID required'}), 400
    
    optimization = hospitality_engine.optimize_menu_engineering(business_id)
    return jsonify(optimization)

@app.route('/hospitality/api/staff-scheduling', methods=['POST'])
def optimize_scheduling():
    """API endpoint for staff scheduling optimization"""
    
    data = request.get_json()
    business_id = data.get('business_id')
    
    if not business_id:
        return jsonify({'error': 'Business ID required'}), 400
    
    optimization = hospitality_engine.optimize_staff_scheduling(business_id)
    return jsonify(optimization)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if HospitalityBusiness.query.count() == 0:
        sample_business = HospitalityBusiness(
            business_id='HOSP_DEMO_001',
            business_name='Demo Bistro & Grill',
            business_type=BusinessType.RESTAURANT,
            address='Culinary District',
            seating_capacity=85,
            daily_covers=180,
            average_ticket_size=32.50,
            customer_satisfaction=4.3
        )
        
        db.session.add(sample_business)
        db.session.commit()
        logger.info("Sample hospitality data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5027, debug=True)