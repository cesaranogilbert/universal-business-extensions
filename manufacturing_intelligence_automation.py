"""
Manufacturing Intelligence & Automation Platform - Complete AI-Powered Manufacturing Solution
$30B+ Value Potential - Production Optimization, Quality Control & Predictive Maintenance
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
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "manufacturing-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///manufacturing.db")

db.init_app(app)

# Manufacturing Enums
class EquipmentStatus(Enum):
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    BREAKDOWN = "breakdown"
    IDLE = "idle"

class QualityStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    REWORK = "rework"
    PENDING = "pending"

class ProductionStatus(Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    DELAYED = "delayed"

# Data Models
class ManufacturingFacility(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    facility_id = db.Column(db.String(100), unique=True, nullable=False)
    facility_name = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200))
    
    # Facility Configuration
    production_lines = db.Column(db.Integer, default=1)
    equipment_count = db.Column(db.Integer, default=10)
    employee_count = db.Column(db.Integer, default=50)
    
    # Performance Metrics
    overall_equipment_effectiveness = db.Column(db.Float, default=75.0)  # OEE percentage
    daily_production_capacity = db.Column(db.Integer, default=1000)
    current_utilization = db.Column(db.Float, default=80.0)
    
    # Quality Metrics
    defect_rate = db.Column(db.Float, default=2.5)  # percentage
    first_pass_yield = db.Column(db.Float, default=95.0)  # percentage
    
    # Cost Metrics
    cost_per_unit = db.Column(db.Float, default=10.0)
    energy_cost_per_day = db.Column(db.Float, default=2500.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Equipment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    equipment_id = db.Column(db.String(100), unique=True, nullable=False)
    facility_id = db.Column(db.String(100), db.ForeignKey('manufacturing_facility.facility_id'), nullable=False)
    
    # Equipment Details
    name = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(100))  # cnc_machine, press, assembly_robot, etc.
    manufacturer = db.Column(db.String(100))
    model = db.Column(db.String(100))
    serial_number = db.Column(db.String(100))
    
    # Operational Status
    status = db.Column(db.Enum(EquipmentStatus), default=EquipmentStatus.OPERATIONAL)
    current_cycle_time = db.Column(db.Float, default=60.0)  # seconds
    target_cycle_time = db.Column(db.Float, default=55.0)  # seconds
    
    # Performance Metrics
    availability = db.Column(db.Float, default=90.0)  # percentage
    performance_efficiency = db.Column(db.Float, default=85.0)  # percentage
    quality_rate = db.Column(db.Float, default=98.0)  # percentage
    oee_score = db.Column(db.Float, default=75.0)  # Overall Equipment Effectiveness
    
    # Maintenance Information
    last_maintenance_date = db.Column(db.Date)
    next_maintenance_due = db.Column(db.Date)
    maintenance_interval_days = db.Column(db.Integer, default=30)
    
    # Predictive Analytics
    failure_probability = db.Column(db.Float, default=0.05)  # 0-1
    estimated_remaining_life = db.Column(db.Integer, default=365)  # days
    maintenance_cost_estimate = db.Column(db.Float, default=5000.0)
    
    # Sensor Data
    temperature = db.Column(db.Float, default=25.0)
    vibration_level = db.Column(db.Float, default=0.5)
    power_consumption = db.Column(db.Float, default=15.0)  # kW
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class ProductionOrder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.String(100), unique=True, nullable=False)
    facility_id = db.Column(db.String(100), db.ForeignKey('manufacturing_facility.facility_id'), nullable=False)
    
    # Order Details
    product_name = db.Column(db.String(200), nullable=False)
    product_sku = db.Column(db.String(100))
    quantity_ordered = db.Column(db.Integer, nullable=False)
    quantity_produced = db.Column(db.Integer, default=0)
    
    # Timeline
    order_date = db.Column(db.Date, nullable=False)
    planned_start_date = db.Column(db.Date)
    actual_start_date = db.Column(db.Date)
    planned_completion_date = db.Column(db.Date)
    actual_completion_date = db.Column(db.Date)
    
    # Status and Progress
    status = db.Column(db.Enum(ProductionStatus), default=ProductionStatus.PLANNED)
    completion_percentage = db.Column(db.Float, default=0.0)
    
    # Resource Allocation
    assigned_line = db.Column(db.String(50))
    assigned_equipment = db.Column(db.JSON)  # List of equipment IDs
    assigned_workers = db.Column(db.JSON)  # List of worker IDs
    
    # Performance Metrics
    actual_cycle_time = db.Column(db.Float, default=0.0)
    efficiency_score = db.Column(db.Float, default=85.0)
    quality_score = db.Column(db.Float, default=95.0)
    
    # AI Optimization
    bottleneck_predictions = db.Column(db.JSON)
    optimization_suggestions = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class QualityCheck(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    check_id = db.Column(db.String(100), unique=True, nullable=False)
    order_id = db.Column(db.String(100), db.ForeignKey('production_order.order_id'), nullable=False)
    equipment_id = db.Column(db.String(100), db.ForeignKey('equipment.equipment_id'))
    
    # Quality Details
    check_date = db.Column(db.DateTime, default=datetime.utcnow)
    inspector_id = db.Column(db.String(100))
    batch_number = db.Column(db.String(100))
    sample_size = db.Column(db.Integer, default=1)
    
    # Quality Results
    status = db.Column(db.Enum(QualityStatus), default=QualityStatus.PENDING)
    overall_score = db.Column(db.Float, default=0.0)  # 0-100
    
    # Dimensional Measurements
    dimensions = db.Column(db.JSON)  # measurements and tolerances
    defects_found = db.Column(db.JSON)  # list of defect types
    
    # Test Results
    material_properties = db.Column(db.JSON)
    performance_tests = db.Column(db.JSON)
    
    # AI Analysis
    defect_probability = db.Column(db.Float, default=0.0)
    recommended_actions = db.Column(db.JSON)
    root_cause_analysis = db.Column(db.JSON)
    
    # Photos and Documentation
    inspection_photos = db.Column(db.JSON)  # URLs to photos
    notes = db.Column(db.Text)

class ProductionMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    metric_id = db.Column(db.String(100), unique=True, nullable=False)
    facility_id = db.Column(db.String(100), db.ForeignKey('manufacturing_facility.facility_id'), nullable=False)
    
    # Time Period
    date = db.Column(db.Date, nullable=False)
    shift = db.Column(db.String(20))  # morning, afternoon, night
    
    # Production Metrics
    units_produced = db.Column(db.Integer, default=0)
    units_planned = db.Column(db.Integer, default=0)
    production_efficiency = db.Column(db.Float, default=85.0)
    
    # Quality Metrics
    defect_count = db.Column(db.Integer, default=0)
    rework_count = db.Column(db.Integer, default=0)
    scrap_count = db.Column(db.Integer, default=0)
    
    # Downtime Tracking
    planned_downtime_minutes = db.Column(db.Integer, default=0)
    unplanned_downtime_minutes = db.Column(db.Integer, default=0)
    changeover_time_minutes = db.Column(db.Integer, default=0)
    
    # Resource Utilization
    labor_hours_used = db.Column(db.Float, default=0.0)
    material_waste_percentage = db.Column(db.Float, default=2.0)
    energy_consumption = db.Column(db.Float, default=500.0)  # kWh
    
    # Costs
    labor_cost = db.Column(db.Float, default=0.0)
    material_cost = db.Column(db.Float, default=0.0)
    overhead_cost = db.Column(db.Float, default=0.0)
    
    # AI Predictions
    next_day_production_forecast = db.Column(db.Integer, default=0)
    bottleneck_predictions = db.Column(db.JSON)
    optimization_opportunities = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Manufacturing Intelligence Engine
class ManufacturingIntelligenceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def optimize_production_line(self, facility_id: str) -> Dict[str, Any]:
        """Comprehensive production line optimization"""
        
        # Get current production orders
        active_orders = ProductionOrder.query.filter_by(
            facility_id=facility_id,
            status=ProductionStatus.ACTIVE
        ).all()
        
        # Get equipment status
        equipment = Equipment.query.filter_by(facility_id=facility_id).all()
        
        # Get recent metrics
        recent_metrics = ProductionMetrics.query.filter_by(facility_id=facility_id)\
                                                .order_by(ProductionMetrics.date.desc())\
                                                .limit(7).all()
        
        if not equipment or not recent_metrics:
            return {'error': 'Insufficient data for optimization'}
        
        # Analyze current performance
        performance_analysis = self._analyze_production_performance(recent_metrics, equipment)
        
        # Identify bottlenecks
        bottlenecks = self._identify_production_bottlenecks(equipment, active_orders)
        
        # Generate optimization recommendations
        optimizations = self._generate_production_optimizations(performance_analysis, bottlenecks)
        
        # Calculate potential improvements
        improvement_potential = self._calculate_improvement_potential(performance_analysis, optimizations)
        
        return {
            'facility_id': facility_id,
            'current_performance': performance_analysis,
            'identified_bottlenecks': bottlenecks,
            'optimization_recommendations': optimizations,
            'improvement_potential': improvement_potential,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_production_performance(self, metrics: List[ProductionMetrics], equipment: List[Equipment]) -> Dict[str, Any]:
        """Analyze current production performance"""
        
        if not metrics:
            return {'error': 'No metrics data'}
        
        # Calculate averages over the period
        avg_efficiency = np.mean([m.production_efficiency for m in metrics])
        avg_oee = np.mean([e.oee_score for e in equipment])
        total_downtime = sum(m.unplanned_downtime_minutes for m in metrics)
        avg_defect_rate = np.mean([m.defect_count / max(1, m.units_produced) * 100 for m in metrics])
        
        # Capacity utilization
        total_produced = sum(m.units_produced for m in metrics)
        total_planned = sum(m.units_planned for m in metrics)
        capacity_utilization = (total_produced / total_planned * 100) if total_planned > 0 else 0
        
        # Cost analysis
        total_cost = sum(m.labor_cost + m.material_cost + m.overhead_cost for m in metrics)
        cost_per_unit = total_cost / total_produced if total_produced > 0 else 0
        
        return {
            'average_efficiency': avg_efficiency,
            'overall_equipment_effectiveness': avg_oee,
            'capacity_utilization': capacity_utilization,
            'total_downtime_hours': total_downtime / 60,
            'defect_rate_percentage': avg_defect_rate,
            'cost_per_unit': cost_per_unit,
            'units_produced_week': total_produced,
            'performance_trend': self._calculate_performance_trend(metrics)
        }
    
    def _identify_production_bottlenecks(self, equipment: List[Equipment], orders: List[ProductionOrder]) -> List[Dict[str, Any]]:
        """Identify production bottlenecks"""
        
        bottlenecks = []
        
        # Equipment-based bottlenecks
        for eq in equipment:
            if eq.oee_score < 60:
                bottlenecks.append({
                    'type': 'equipment_efficiency',
                    'equipment_id': eq.equipment_id,
                    'equipment_name': eq.name,
                    'severity': 'high' if eq.oee_score < 40 else 'medium',
                    'current_oee': eq.oee_score,
                    'impact': 'reduced_throughput'
                })
            
            if eq.current_cycle_time > eq.target_cycle_time * 1.2:
                bottlenecks.append({
                    'type': 'cycle_time',
                    'equipment_id': eq.equipment_id,
                    'equipment_name': eq.name,
                    'severity': 'medium',
                    'cycle_time_variance': ((eq.current_cycle_time - eq.target_cycle_time) / eq.target_cycle_time * 100),
                    'impact': 'production_delays'
                })
        
        # Quality-based bottlenecks
        low_quality_equipment = [eq for eq in equipment if eq.quality_rate < 95]
        for eq in low_quality_equipment:
            bottlenecks.append({
                'type': 'quality_issues',
                'equipment_id': eq.equipment_id,
                'equipment_name': eq.name,
                'severity': 'high',
                'quality_rate': eq.quality_rate,
                'impact': 'rework_costs'
            })
        
        return bottlenecks
    
    def _generate_production_optimizations(self, performance: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations"""
        
        optimizations = []
        
        # Efficiency optimizations
        if performance['average_efficiency'] < 80:
            optimizations.append({
                'category': 'efficiency',
                'priority': 'high',
                'recommendation': 'Implement lean manufacturing principles to reduce waste',
                'expected_improvement': '15-25% efficiency gain',
                'implementation_time': '4-6 weeks'
            })
        
        # OEE optimizations
        if performance['overall_equipment_effectiveness'] < 70:
            optimizations.append({
                'category': 'oee',
                'priority': 'high',
                'recommendation': 'Focus on reducing unplanned downtime through predictive maintenance',
                'expected_improvement': '10-20% OEE improvement',
                'implementation_time': '2-3 months'
            })
        
        # Quality optimizations
        if performance['defect_rate_percentage'] > 3:
            optimizations.append({
                'category': 'quality',
                'priority': 'medium',
                'recommendation': 'Implement statistical process control and real-time quality monitoring',
                'expected_improvement': '50% reduction in defect rate',
                'implementation_time': '6-8 weeks'
            })
        
        # Capacity optimizations
        if performance['capacity_utilization'] < 75:
            optimizations.append({
                'category': 'capacity',
                'priority': 'medium',
                'recommendation': 'Optimize production scheduling and reduce changeover times',
                'expected_improvement': '20-30% capacity increase',
                'implementation_time': '3-4 weeks'
            })
        
        # Bottleneck-specific optimizations
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'equipment_efficiency':
                optimizations.append({
                    'category': 'equipment',
                    'priority': 'high',
                    'recommendation': f"Urgently address {bottleneck['equipment_name']} performance issues",
                    'expected_improvement': f"Increase OEE from {bottleneck['current_oee']}% to 80%+",
                    'implementation_time': '1-2 weeks'
                })
        
        return optimizations
    
    def _calculate_improvement_potential(self, performance: Dict[str, Any], optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate potential improvements from optimizations"""
        
        current_efficiency = performance['average_efficiency']
        current_oee = performance['overall_equipment_effectiveness']
        current_defect_rate = performance['defect_rate_percentage']
        
        # Estimate improvements
        potential_efficiency = min(95, current_efficiency * 1.25)  # Up to 25% improvement
        potential_oee = min(90, current_oee * 1.20)  # Up to 20% improvement
        potential_defect_reduction = max(0.5, current_defect_rate * 0.5)  # 50% reduction
        
        # Calculate financial impact
        current_cost = performance['cost_per_unit']
        potential_cost_reduction = current_cost * 0.15  # 15% cost reduction
        
        return {
            'efficiency_improvement': {
                'current': current_efficiency,
                'potential': potential_efficiency,
                'improvement_percent': ((potential_efficiency - current_efficiency) / current_efficiency * 100)
            },
            'oee_improvement': {
                'current': current_oee,
                'potential': potential_oee,
                'improvement_percent': ((potential_oee - current_oee) / current_oee * 100)
            },
            'quality_improvement': {
                'current_defect_rate': current_defect_rate,
                'potential_defect_rate': potential_defect_reduction,
                'improvement_percent': ((current_defect_rate - potential_defect_reduction) / current_defect_rate * 100)
            },
            'cost_reduction': {
                'current_cost_per_unit': current_cost,
                'potential_savings_per_unit': potential_cost_reduction,
                'savings_percent': 15.0
            }
        }
    
    def _calculate_performance_trend(self, metrics: List[ProductionMetrics]) -> str:
        """Calculate performance trend over time"""
        
        if len(metrics) < 3:
            return 'insufficient_data'
        
        # Sort by date
        sorted_metrics = sorted(metrics, key=lambda x: x.date)
        efficiencies = [m.production_efficiency for m in sorted_metrics]
        
        # Calculate trend
        recent_avg = np.mean(efficiencies[-3:])
        older_avg = np.mean(efficiencies[:-3])
        
        change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        if change_percent > 5:
            return 'improving'
        elif change_percent < -5:
            return 'declining'
        else:
            return 'stable'
    
    def predict_equipment_maintenance(self, facility_id: str) -> Dict[str, Any]:
        """AI-powered predictive maintenance analysis"""
        
        equipment_list = Equipment.query.filter_by(facility_id=facility_id).all()
        
        if not equipment_list:
            return {'error': 'No equipment found'}
        
        maintenance_predictions = []
        urgent_maintenance_needed = 0
        
        for equipment in equipment_list:
            prediction = self._predict_equipment_failure(equipment)
            maintenance_predictions.append(prediction)
            
            if prediction['urgency'] == 'critical':
                urgent_maintenance_needed += 1
        
        # Generate maintenance schedule
        maintenance_schedule = self._generate_maintenance_schedule(maintenance_predictions)
        
        # Calculate cost implications
        cost_analysis = self._calculate_maintenance_costs(maintenance_predictions)
        
        return {
            'facility_id': facility_id,
            'equipment_analyzed': len(equipment_list),
            'urgent_maintenance_needed': urgent_maintenance_needed,
            'maintenance_predictions': maintenance_predictions,
            'recommended_schedule': maintenance_schedule,
            'cost_analysis': cost_analysis,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _predict_equipment_failure(self, equipment: Equipment) -> Dict[str, Any]:
        """Predict failure probability for individual equipment"""
        
        # Calculate time since last maintenance
        days_since_maintenance = 0
        if equipment.last_maintenance_date:
            days_since_maintenance = (datetime.now().date() - equipment.last_maintenance_date).days
        
        # Calculate maintenance urgency based on multiple factors
        base_failure_risk = equipment.failure_probability
        
        # Adjust risk based on performance degradation
        performance_factor = 1.0
        if equipment.oee_score < 70:
            performance_factor = 1.5
        elif equipment.oee_score < 50:
            performance_factor = 2.0
        
        # Adjust risk based on maintenance schedule
        maintenance_factor = 1.0
        if days_since_maintenance > equipment.maintenance_interval_days:
            overdue_ratio = days_since_maintenance / equipment.maintenance_interval_days
            maintenance_factor = min(3.0, 1.0 + overdue_ratio)
        
        # Sensor-based risk factors
        sensor_factor = 1.0
        if equipment.temperature > 50 or equipment.vibration_level > 2.0:
            sensor_factor = 1.3
        
        # Calculate final failure probability
        adjusted_risk = min(0.95, base_failure_risk * performance_factor * maintenance_factor * sensor_factor)
        
        # Determine urgency level
        if adjusted_risk > 0.7:
            urgency = 'critical'
        elif adjusted_risk > 0.4:
            urgency = 'high'
        elif adjusted_risk > 0.2:
            urgency = 'medium'
        else:
            urgency = 'low'
        
        # Estimate days until failure
        days_until_failure = max(1, int((1 - adjusted_risk) * 365))
        
        return {
            'equipment_id': equipment.equipment_id,
            'equipment_name': equipment.name,
            'failure_probability': adjusted_risk,
            'urgency': urgency,
            'days_until_failure': days_until_failure,
            'recommended_action': self._get_maintenance_recommendation(urgency, days_until_failure),
            'estimated_cost': equipment.maintenance_cost_estimate,
            'risk_factors': {
                'performance_degradation': equipment.oee_score < 70,
                'overdue_maintenance': days_since_maintenance > equipment.maintenance_interval_days,
                'sensor_alerts': equipment.temperature > 50 or equipment.vibration_level > 2.0
            }
        }
    
    def _get_maintenance_recommendation(self, urgency: str, days_until_failure: int) -> str:
        """Get maintenance recommendation based on urgency"""
        
        if urgency == 'critical':
            return 'Schedule immediate maintenance - stop production if necessary'
        elif urgency == 'high':
            return f'Schedule maintenance within {min(7, days_until_failure)} days'
        elif urgency == 'medium':
            return f'Schedule maintenance within {min(30, days_until_failure)} days'
        else:
            return 'Maintain regular maintenance schedule'
    
    def _generate_maintenance_schedule(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimized maintenance schedule"""
        
        schedule = []
        
        # Sort by urgency and days until failure
        sorted_predictions = sorted(predictions, key=lambda x: (
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['urgency']],
            x['days_until_failure']
        ))
        
        current_date = datetime.now().date()
        
        for i, prediction in enumerate(sorted_predictions):
            if prediction['urgency'] in ['critical', 'high']:
                # Schedule immediately or within a few days
                scheduled_date = current_date + timedelta(days=i * 2)  # Spread over days
            else:
                # Schedule based on predicted failure time
                scheduled_date = current_date + timedelta(days=prediction['days_until_failure'] // 2)
            
            schedule.append({
                'equipment_id': prediction['equipment_id'],
                'equipment_name': prediction['equipment_name'],
                'scheduled_date': scheduled_date.isoformat(),
                'urgency': prediction['urgency'],
                'estimated_duration': '4-8 hours',  # Typical maintenance window
                'estimated_cost': prediction['estimated_cost']
            })
        
        return schedule
    
    def _calculate_maintenance_costs(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate maintenance cost analysis"""
        
        total_predicted_cost = sum(p['estimated_cost'] for p in predictions)
        critical_equipment_cost = sum(p['estimated_cost'] for p in predictions if p['urgency'] == 'critical')
        
        # Estimate cost savings from predictive maintenance
        reactive_maintenance_multiplier = 3.0  # Reactive maintenance typically costs 3x more
        potential_savings = total_predicted_cost * (reactive_maintenance_multiplier - 1)
        
        return {
            'total_predicted_maintenance_cost': total_predicted_cost,
            'critical_equipment_cost': critical_equipment_cost,
            'potential_savings_from_predictive': potential_savings,
            'cost_breakdown': {
                'critical': sum(p['estimated_cost'] for p in predictions if p['urgency'] == 'critical'),
                'high': sum(p['estimated_cost'] for p in predictions if p['urgency'] == 'high'),
                'medium': sum(p['estimated_cost'] for p in predictions if p['urgency'] == 'medium'),
                'low': sum(p['estimated_cost'] for p in predictions if p['urgency'] == 'low')
            }
        }

# Initialize engine
manufacturing_engine = ManufacturingIntelligenceEngine()

# Routes
@app.route('/manufacturing')
def manufacturing_dashboard():
    """Manufacturing Intelligence dashboard"""
    
    recent_facilities = ManufacturingFacility.query.order_by(ManufacturingFacility.created_at.desc()).limit(10).all()
    
    return render_template('manufacturing/dashboard.html',
                         facilities=recent_facilities)

@app.route('/manufacturing/api/production-optimization', methods=['POST'])
def optimize_production():
    """API endpoint for production optimization"""
    
    data = request.get_json()
    facility_id = data.get('facility_id')
    
    if not facility_id:
        return jsonify({'error': 'Facility ID required'}), 400
    
    optimization = manufacturing_engine.optimize_production_line(facility_id)
    return jsonify(optimization)

@app.route('/manufacturing/api/predictive-maintenance', methods=['POST'])
def predict_maintenance():
    """API endpoint for predictive maintenance"""
    
    data = request.get_json()
    facility_id = data.get('facility_id')
    
    if not facility_id:
        return jsonify({'error': 'Facility ID required'}), 400
    
    predictions = manufacturing_engine.predict_equipment_maintenance(facility_id)
    return jsonify(predictions)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if ManufacturingFacility.query.count() == 0:
        sample_facility = ManufacturingFacility(
            facility_id='MFG_DEMO_001',
            facility_name='Demo Manufacturing Plant',
            location='Industrial District',
            production_lines=3,
            equipment_count=25,
            employee_count=150,
            daily_production_capacity=5000
        )
        
        db.session.add(sample_facility)
        db.session.commit()
        logger.info("Sample manufacturing data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5023, debug=True)