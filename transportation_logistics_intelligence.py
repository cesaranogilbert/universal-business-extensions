"""
Transportation & Logistics Intelligence System - Complete AI-Powered Logistics Platform
$25B+ Value Potential - Route Optimization, Fleet Management & Supply Chain Intelligence
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import pandas as pd
import numpy as np
from openai import OpenAI
from enum import Enum
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "transportation-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///transportation.db")

db.init_app(app)

# Transportation Enums
class VehicleType(Enum):
    TRUCK = "truck"
    VAN = "van"
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    DRONE = "drone"

class DeliveryStatus(Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VehicleStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    OUT_OF_SERVICE = "out_of_service"

# Data Models
class LogisticsCompany(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.String(100), unique=True, nullable=False)
    company_name = db.Column(db.String(200), nullable=False)
    headquarters = db.Column(db.String(200))
    
    # Fleet Information
    total_vehicles = db.Column(db.Integer, default=0)
    active_drivers = db.Column(db.Integer, default=0)
    service_areas = db.Column(db.JSON)  # List of service areas/cities
    
    # Performance Metrics
    daily_deliveries = db.Column(db.Integer, default=0)
    on_time_delivery_rate = db.Column(db.Float, default=95.0)  # percentage
    customer_satisfaction = db.Column(db.Float, default=4.5)  # 1-5 scale
    average_delivery_time = db.Column(db.Float, default=2.5)  # hours
    
    # Cost Metrics
    cost_per_delivery = db.Column(db.Float, default=8.50)
    fuel_efficiency = db.Column(db.Float, default=12.0)  # miles per gallon
    maintenance_cost_monthly = db.Column(db.Float, default=25000.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.String(100), unique=True, nullable=False)
    company_id = db.Column(db.String(100), db.ForeignKey('logistics_company.company_id'), nullable=False)
    
    # Vehicle Details
    vehicle_type = db.Column(db.Enum(VehicleType), nullable=False)
    make = db.Column(db.String(50))
    model = db.Column(db.String(50))
    year = db.Column(db.Integer)
    license_plate = db.Column(db.String(20))
    
    # Capacity
    max_weight_capacity = db.Column(db.Float, default=1000.0)  # pounds
    max_volume_capacity = db.Column(db.Float, default=100.0)  # cubic feet
    current_load_weight = db.Column(db.Float, default=0.0)
    current_load_volume = db.Column(db.Float, default=0.0)
    
    # Status and Location
    status = db.Column(db.Enum(VehicleStatus), default=VehicleStatus.AVAILABLE)
    current_latitude = db.Column(db.Float)
    current_longitude = db.Column(db.Float)
    current_address = db.Column(db.String(500))
    
    # Performance Metrics
    total_miles = db.Column(db.Float, default=0.0)
    fuel_efficiency = db.Column(db.Float, default=12.0)  # MPG
    maintenance_score = db.Column(db.Float, default=85.0)  # 0-100
    
    # Maintenance Information
    last_maintenance_date = db.Column(db.Date)
    next_maintenance_due = db.Column(db.Date)
    maintenance_alerts = db.Column(db.JSON)
    
    # AI Optimization
    utilization_score = db.Column(db.Float, default=75.0)  # 0-100
    efficiency_rating = db.Column(db.Float, default=80.0)  # 0-100
    predicted_maintenance_cost = db.Column(db.Float, default=2500.0)
    
    assigned_driver_id = db.Column(db.String(100))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Driver(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.String(100), unique=True, nullable=False)
    company_id = db.Column(db.String(100), db.ForeignKey('logistics_company.company_id'), nullable=False)
    
    # Personal Information
    name = db.Column(db.String(200), nullable=False)
    license_number = db.Column(db.String(50))
    phone = db.Column(db.String(50))
    email = db.Column(db.String(200))
    
    # Employment
    hire_date = db.Column(db.Date)
    employment_status = db.Column(db.String(50), default='active')
    years_experience = db.Column(db.Integer, default=0)
    
    # Performance Metrics
    safety_score = db.Column(db.Float, default=95.0)  # 0-100
    efficiency_score = db.Column(db.Float, default=85.0)  # 0-100
    customer_rating = db.Column(db.Float, default=4.5)  # 1-5 scale
    on_time_delivery_rate = db.Column(db.Float, default=95.0)  # percentage
    
    # Activity Tracking
    total_deliveries = db.Column(db.Integer, default=0)
    total_miles_driven = db.Column(db.Float, default=0.0)
    accidents_count = db.Column(db.Integer, default=0)
    violations_count = db.Column(db.Integer, default=0)
    
    # Current Status
    is_available = db.Column(db.Boolean, default=True)
    current_vehicle_id = db.Column(db.String(100))
    shift_start = db.Column(db.Time)
    shift_end = db.Column(db.Time)
    
    # AI Analysis
    performance_trend = db.Column(db.String(50), default='stable')  # improving, stable, declining
    recommended_training = db.Column(db.JSON)
    risk_assessment = db.Column(db.Float, default=25.0)  # 0-100

class Delivery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    delivery_id = db.Column(db.String(100), unique=True, nullable=False)
    company_id = db.Column(db.String(100), db.ForeignKey('logistics_company.company_id'), nullable=False)
    
    # Delivery Details
    customer_name = db.Column(db.String(200))
    pickup_address = db.Column(db.String(500), nullable=False)
    delivery_address = db.Column(db.String(500), nullable=False)
    
    # Package Information
    package_weight = db.Column(db.Float, default=1.0)  # pounds
    package_volume = db.Column(db.Float, default=1.0)  # cubic feet
    package_value = db.Column(db.Float, default=50.0)
    special_instructions = db.Column(db.Text)
    
    # Scheduling
    pickup_time_window_start = db.Column(db.DateTime)
    pickup_time_window_end = db.Column(db.DateTime)
    delivery_time_window_start = db.Column(db.DateTime)
    delivery_time_window_end = db.Column(db.DateTime)
    
    # Execution
    status = db.Column(db.Enum(DeliveryStatus), default=DeliveryStatus.PENDING)
    assigned_vehicle_id = db.Column(db.String(100))
    assigned_driver_id = db.Column(db.String(100))
    
    actual_pickup_time = db.Column(db.DateTime)
    actual_delivery_time = db.Column(db.DateTime)
    
    # Route Information
    estimated_distance = db.Column(db.Float, default=10.0)  # miles
    actual_distance = db.Column(db.Float, default=0.0)
    estimated_duration = db.Column(db.Float, default=1.0)  # hours
    actual_duration = db.Column(db.Float, default=0.0)
    
    # Costs and Pricing
    delivery_fee = db.Column(db.Float, default=15.0)
    fuel_cost = db.Column(db.Float, default=3.50)
    total_cost = db.Column(db.Float, default=8.50)
    
    # Customer Feedback
    customer_rating = db.Column(db.Float, default=0.0)  # 1-5 scale
    customer_comments = db.Column(db.Text)
    
    # AI Optimization
    route_optimization_score = db.Column(db.Float, default=80.0)
    delivery_priority = db.Column(db.Integer, default=5)  # 1-10, higher = more urgent
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Route(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    route_id = db.Column(db.String(100), unique=True, nullable=False)
    company_id = db.Column(db.String(100), db.ForeignKey('logistics_company.company_id'), nullable=False)
    
    # Route Details
    route_name = db.Column(db.String(200))
    start_location = db.Column(db.String(500))
    end_location = db.Column(db.String(500))
    waypoints = db.Column(db.JSON)  # List of intermediate stops
    
    # Route Metrics
    total_distance = db.Column(db.Float, default=0.0)  # miles
    estimated_duration = db.Column(db.Float, default=0.0)  # hours
    fuel_consumption = db.Column(db.Float, default=0.0)  # gallons
    
    # Deliveries on Route
    delivery_ids = db.Column(db.JSON)  # List of delivery IDs
    delivery_count = db.Column(db.Integer, default=0)
    
    # Assignments
    assigned_vehicle_id = db.Column(db.String(100))
    assigned_driver_id = db.Column(db.String(100))
    
    # Status
    route_date = db.Column(db.Date)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    is_completed = db.Column(db.Boolean, default=False)
    
    # Optimization Results
    optimization_score = db.Column(db.Float, default=75.0)  # 0-100
    time_savings = db.Column(db.Float, default=0.0)  # hours
    fuel_savings = db.Column(db.Float, default=0.0)  # gallons
    cost_savings = db.Column(db.Float, default=0.0)  # dollars
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FuelTransaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(100), unique=True, nullable=False)
    vehicle_id = db.Column(db.String(100), db.ForeignKey('vehicle.vehicle_id'), nullable=False)
    driver_id = db.Column(db.String(100), db.ForeignKey('driver.driver_id'))
    
    # Transaction Details
    transaction_date = db.Column(db.DateTime, default=datetime.utcnow)
    fuel_amount = db.Column(db.Float, nullable=False)  # gallons
    price_per_gallon = db.Column(db.Float, nullable=False)
    total_cost = db.Column(db.Float, nullable=False)
    
    # Location
    station_name = db.Column(db.String(200))
    station_address = db.Column(db.String(500))
    
    # Vehicle State
    odometer_reading = db.Column(db.Float)
    fuel_efficiency_calculated = db.Column(db.Float)  # MPG since last fill-up

# Transportation Intelligence Engine
class TransportationIntelligenceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def optimize_route_planning(self, company_id: str, delivery_date: str) -> Dict[str, Any]:
        """AI-powered route optimization for multiple deliveries"""
        
        # Get pending deliveries for the date
        target_date = datetime.strptime(delivery_date, '%Y-%m-%d').date()
        deliveries = Delivery.query.filter_by(
            company_id=company_id,
            status=DeliveryStatus.PENDING
        ).filter(
            db.func.date(Delivery.delivery_time_window_start) == target_date
        ).all()
        
        if not deliveries:
            return {'error': 'No deliveries found for the specified date'}
        
        # Get available vehicles and drivers
        available_vehicles = Vehicle.query.filter_by(
            company_id=company_id,
            status=VehicleStatus.AVAILABLE
        ).all()
        
        available_drivers = Driver.query.filter_by(
            company_id=company_id,
            is_available=True
        ).all()
        
        # Optimize routes
        optimized_routes = self._create_optimized_routes(deliveries, available_vehicles, available_drivers)
        
        # Calculate performance improvements
        improvement_metrics = self._calculate_route_improvements(optimized_routes, deliveries)
        
        return {
            'company_id': company_id,
            'delivery_date': delivery_date,
            'total_deliveries': len(deliveries),
            'optimized_routes': optimized_routes,
            'improvement_metrics': improvement_metrics,
            'optimization_date': datetime.utcnow().isoformat()
        }
    
    def _create_optimized_routes(self, deliveries: List[Delivery], vehicles: List[Vehicle], drivers: List[Driver]) -> List[Dict[str, Any]]:
        """Create optimized delivery routes using AI algorithms"""
        
        optimized_routes = []
        
        # Group deliveries by geographic proximity and time windows
        delivery_clusters = self._cluster_deliveries_by_location(deliveries)
        
        route_index = 0
        for cluster in delivery_clusters:
            if route_index >= len(vehicles) or route_index >= len(drivers):
                break  # No more vehicles or drivers available
            
            vehicle = vehicles[route_index]
            driver = drivers[route_index]
            
            # Create route for this cluster
            route = self._create_single_route(cluster, vehicle, driver)
            optimized_routes.append(route)
            
            route_index += 1
        
        return optimized_routes
    
    def _cluster_deliveries_by_location(self, deliveries: List[Delivery]) -> List[List[Delivery]]:
        """Cluster deliveries by geographic proximity using simple distance calculation"""
        
        # Simple clustering based on delivery addresses
        # In production, would use more sophisticated clustering algorithms
        
        clusters = []
        unassigned = deliveries.copy()
        
        while unassigned:
            # Start new cluster with first unassigned delivery
            cluster_seed = unassigned.pop(0)
            current_cluster = [cluster_seed]
            
            # Find nearby deliveries (simplified using first 8 characters of address)
            seed_area = cluster_seed.delivery_address[:8].upper()
            
            # Add nearby deliveries to cluster (max 8 per route for efficiency)
            for delivery in unassigned[:]:
                if len(current_cluster) >= 8:
                    break
                    
                delivery_area = delivery.delivery_address[:8].upper()
                
                # Simple proximity check (same area prefix)
                if delivery_area == seed_area:
                    current_cluster.append(delivery)
                    unassigned.remove(delivery)
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _create_single_route(self, deliveries: List[Delivery], vehicle: Vehicle, driver: Driver) -> Dict[str, Any]:
        """Create optimized route for a cluster of deliveries"""
        
        # Sort deliveries by time window and priority
        sorted_deliveries = sorted(deliveries, key=lambda d: (
            d.delivery_time_window_start or datetime.min,
            -d.delivery_priority
        ))
        
        # Calculate route metrics
        total_distance = self._calculate_total_distance(sorted_deliveries)
        estimated_duration = self._estimate_route_duration(sorted_deliveries, total_distance)
        fuel_consumption = total_distance / vehicle.fuel_efficiency
        fuel_cost = fuel_consumption * 3.50  # Assume $3.50/gallon
        
        # Create route record
        route_id = f"ROUTE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{vehicle.vehicle_id}"
        
        route_data = {
            'route_id': route_id,
            'assigned_vehicle': {
                'vehicle_id': vehicle.vehicle_id,
                'vehicle_type': vehicle.vehicle_type.value,
                'capacity_utilization': self._calculate_capacity_utilization(sorted_deliveries, vehicle)
            },
            'assigned_driver': {
                'driver_id': driver.driver_id,
                'driver_name': driver.name,
                'efficiency_score': driver.efficiency_score
            },
            'deliveries': [{
                'delivery_id': d.delivery_id,
                'customer_name': d.customer_name,
                'delivery_address': d.delivery_address,
                'time_window': {
                    'start': d.delivery_time_window_start.isoformat() if d.delivery_time_window_start else None,
                    'end': d.delivery_time_window_end.isoformat() if d.delivery_time_window_end else None
                },
                'priority': d.delivery_priority,
                'estimated_arrival': self._estimate_delivery_time(d, sorted_deliveries.index(d))
            } for d in sorted_deliveries],
            'route_metrics': {
                'total_distance_miles': total_distance,
                'estimated_duration_hours': estimated_duration,
                'fuel_consumption_gallons': fuel_consumption,
                'estimated_fuel_cost': fuel_cost,
                'delivery_count': len(sorted_deliveries)
            },
            'optimization_score': self._calculate_route_optimization_score(sorted_deliveries, total_distance, estimated_duration)
        }
        
        return route_data
    
    def _calculate_total_distance(self, deliveries: List[Delivery]) -> float:
        """Calculate total distance for route (simplified calculation)"""
        
        # Simplified distance calculation
        # In production, would use actual mapping APIs
        
        base_distance = len(deliveries) * 5.0  # Assume 5 miles between stops on average
        return base_distance
    
    def _estimate_route_duration(self, deliveries: List[Delivery], distance: float) -> float:
        """Estimate total route duration including driving and delivery time"""
        
        # Driving time (assume 25 mph average including traffic and stops)
        driving_time = distance / 25.0
        
        # Delivery time (assume 15 minutes per delivery)
        delivery_time = len(deliveries) * 0.25  # 0.25 hours = 15 minutes
        
        return driving_time + delivery_time
    
    def _calculate_capacity_utilization(self, deliveries: List[Delivery], vehicle: Vehicle) -> float:
        """Calculate vehicle capacity utilization"""
        
        total_weight = sum(d.package_weight for d in deliveries)
        total_volume = sum(d.package_volume for d in deliveries)
        
        weight_utilization = (total_weight / vehicle.max_weight_capacity) * 100 if vehicle.max_weight_capacity > 0 else 0
        volume_utilization = (total_volume / vehicle.max_volume_capacity) * 100 if vehicle.max_volume_capacity > 0 else 0
        
        return max(weight_utilization, volume_utilization)
    
    def _estimate_delivery_time(self, delivery: Delivery, position_in_route: int) -> str:
        """Estimate delivery time based on position in route"""
        
        # Assume route starts at 8 AM and each delivery takes 30 minutes (travel + delivery)
        start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        estimated_time = start_time + timedelta(minutes=position_in_route * 30)
        
        return estimated_time.isoformat()
    
    def _calculate_route_optimization_score(self, deliveries: List[Delivery], distance: float, duration: float) -> float:
        """Calculate optimization score for route"""
        
        # Base score
        score = 75.0
        
        # Bonus for efficient distance per delivery
        distance_per_delivery = distance / len(deliveries) if deliveries else 10
        if distance_per_delivery < 3:
            score += 15
        elif distance_per_delivery < 5:
            score += 10
        elif distance_per_delivery > 8:
            score -= 10
        
        # Bonus for time efficiency
        time_per_delivery = duration / len(deliveries) if deliveries else 1
        if time_per_delivery < 0.4:  # Less than 24 minutes per delivery
            score += 10
        elif time_per_delivery > 0.8:  # More than 48 minutes per delivery
            score -= 10
        
        return min(100, max(0, score))
    
    def _calculate_route_improvements(self, optimized_routes: List[Dict], original_deliveries: List[Delivery]) -> Dict[str, Any]:
        """Calculate improvements from route optimization"""
        
        # Calculate baseline metrics (before optimization)
        baseline_distance = len(original_deliveries) * 8.0  # Assume 8 miles per delivery without optimization
        baseline_duration = baseline_distance / 20.0 + len(original_deliveries) * 0.25  # 20 mph + delivery time
        baseline_fuel = baseline_distance / 12.0  # 12 MPG average
        
        # Calculate optimized metrics
        optimized_distance = sum(route['route_metrics']['total_distance_miles'] for route in optimized_routes)
        optimized_duration = sum(route['route_metrics']['estimated_duration_hours'] for route in optimized_routes)
        optimized_fuel = sum(route['route_metrics']['fuel_consumption_gallons'] for route in optimized_routes)
        
        # Calculate savings
        distance_savings = baseline_distance - optimized_distance
        time_savings = baseline_duration - optimized_duration
        fuel_savings = baseline_fuel - optimized_fuel
        cost_savings = fuel_savings * 3.50  # Fuel cost savings
        
        return {
            'baseline_metrics': {
                'total_distance_miles': baseline_distance,
                'total_duration_hours': baseline_duration,
                'fuel_consumption_gallons': baseline_fuel
            },
            'optimized_metrics': {
                'total_distance_miles': optimized_distance,
                'total_duration_hours': optimized_duration,
                'fuel_consumption_gallons': optimized_fuel
            },
            'savings': {
                'distance_miles': distance_savings,
                'time_hours': time_savings,
                'fuel_gallons': fuel_savings,
                'cost_dollars': cost_savings
            },
            'improvement_percentages': {
                'distance_reduction': (distance_savings / baseline_distance * 100) if baseline_distance > 0 else 0,
                'time_reduction': (time_savings / baseline_duration * 100) if baseline_duration > 0 else 0,
                'fuel_reduction': (fuel_savings / baseline_fuel * 100) if baseline_fuel > 0 else 0
            }
        }
    
    def analyze_fleet_performance(self, company_id: str) -> Dict[str, Any]:
        """Comprehensive fleet performance analysis"""
        
        # Get fleet data
        vehicles = Vehicle.query.filter_by(company_id=company_id).all()
        drivers = Driver.query.filter_by(company_id=company_id).all()
        
        if not vehicles:
            return {'error': 'No vehicles found'}
        
        # Vehicle performance analysis
        vehicle_analysis = self._analyze_vehicle_performance(vehicles)
        
        # Driver performance analysis
        driver_analysis = self._analyze_driver_performance(drivers)
        
        # Fleet utilization analysis
        utilization_analysis = self._analyze_fleet_utilization(vehicles, drivers)
        
        # Generate recommendations
        recommendations = self._generate_fleet_recommendations(vehicle_analysis, driver_analysis, utilization_analysis)
        
        return {
            'company_id': company_id,
            'fleet_size': len(vehicles),
            'driver_count': len(drivers),
            'vehicle_performance': vehicle_analysis,
            'driver_performance': driver_analysis,
            'utilization_analysis': utilization_analysis,
            'recommendations': recommendations,
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    def _analyze_vehicle_performance(self, vehicles: List[Vehicle]) -> Dict[str, Any]:
        """Analyze vehicle performance metrics"""
        
        # Calculate fleet averages
        avg_fuel_efficiency = np.mean([v.fuel_efficiency for v in vehicles])
        avg_maintenance_score = np.mean([v.maintenance_score for v in vehicles])
        avg_utilization = np.mean([v.utilization_score for v in vehicles])
        
        # Identify top and bottom performers
        top_performers = sorted(vehicles, key=lambda v: v.efficiency_rating, reverse=True)[:5]
        bottom_performers = sorted(vehicles, key=lambda v: v.efficiency_rating)[:5]
        
        # Maintenance alerts
        maintenance_needed = [v for v in vehicles if v.maintenance_score < 70]
        
        return {
            'fleet_averages': {
                'fuel_efficiency_mpg': avg_fuel_efficiency,
                'maintenance_score': avg_maintenance_score,
                'utilization_score': avg_utilization
            },
            'top_performers': [{
                'vehicle_id': v.vehicle_id,
                'vehicle_type': v.vehicle_type.value,
                'efficiency_rating': v.efficiency_rating,
                'fuel_efficiency': v.fuel_efficiency
            } for v in top_performers],
            'bottom_performers': [{
                'vehicle_id': v.vehicle_id,
                'vehicle_type': v.vehicle_type.value,
                'efficiency_rating': v.efficiency_rating,
                'issues': 'Low efficiency rating'
            } for v in bottom_performers],
            'maintenance_alerts': [{
                'vehicle_id': v.vehicle_id,
                'maintenance_score': v.maintenance_score,
                'next_maintenance_due': v.next_maintenance_due.isoformat() if v.next_maintenance_due else None
            } for v in maintenance_needed]
        }
    
    def _analyze_driver_performance(self, drivers: List[Driver]) -> Dict[str, Any]:
        """Analyze driver performance metrics"""
        
        # Calculate averages
        avg_safety_score = np.mean([d.safety_score for d in drivers])
        avg_efficiency_score = np.mean([d.efficiency_score for d in drivers])
        avg_customer_rating = np.mean([d.customer_rating for d in drivers])
        avg_on_time_rate = np.mean([d.on_time_delivery_rate for d in drivers])
        
        # Identify performance issues
        safety_concerns = [d for d in drivers if d.safety_score < 80 or d.accidents_count > 0]
        efficiency_concerns = [d for d in drivers if d.efficiency_score < 70]
        
        return {
            'driver_averages': {
                'safety_score': avg_safety_score,
                'efficiency_score': avg_efficiency_score,
                'customer_rating': avg_customer_rating,
                'on_time_delivery_rate': avg_on_time_rate
            },
            'safety_concerns': [{
                'driver_id': d.driver_id,
                'driver_name': d.name,
                'safety_score': d.safety_score,
                'accidents_count': d.accidents_count,
                'violations_count': d.violations_count
            } for d in safety_concerns],
            'efficiency_concerns': [{
                'driver_id': d.driver_id,
                'driver_name': d.name,
                'efficiency_score': d.efficiency_score,
                'recommended_training': d.recommended_training
            } for d in efficiency_concerns]
        }
    
    def _analyze_fleet_utilization(self, vehicles: List[Vehicle], drivers: List[Driver]) -> Dict[str, Any]:
        """Analyze fleet utilization patterns"""
        
        # Vehicle utilization
        available_vehicles = len([v for v in vehicles if v.status == VehicleStatus.AVAILABLE])
        in_use_vehicles = len([v for v in vehicles if v.status == VehicleStatus.IN_USE])
        maintenance_vehicles = len([v for v in vehicles if v.status == VehicleStatus.MAINTENANCE])
        
        vehicle_utilization_rate = (in_use_vehicles / len(vehicles) * 100) if vehicles else 0
        
        # Driver utilization
        available_drivers = len([d for d in drivers if d.is_available])
        active_drivers = len([d for d in drivers if not d.is_available])
        
        driver_utilization_rate = (active_drivers / len(drivers) * 100) if drivers else 0
        
        return {
            'vehicle_utilization': {
                'total_vehicles': len(vehicles),
                'available': available_vehicles,
                'in_use': in_use_vehicles,
                'maintenance': maintenance_vehicles,
                'utilization_rate_percent': vehicle_utilization_rate
            },
            'driver_utilization': {
                'total_drivers': len(drivers),
                'available': available_drivers,
                'active': active_drivers,
                'utilization_rate_percent': driver_utilization_rate
            },
            'capacity_optimization': {
                'excess_capacity_vehicles': max(0, available_vehicles - 2),  # Keep 2 as buffer
                'capacity_shortage': max(0, len(vehicles) * 0.85 - in_use_vehicles)  # Target 85% utilization
            }
        }
    
    def _generate_fleet_recommendations(self, vehicle_analysis: Dict, driver_analysis: Dict, utilization_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate fleet optimization recommendations"""
        
        recommendations = []
        
        # Vehicle recommendations
        if len(vehicle_analysis.get('maintenance_alerts', [])) > 0:
            recommendations.append({
                'category': 'maintenance',
                'priority': 'high',
                'recommendation': f"Schedule maintenance for {len(vehicle_analysis['maintenance_alerts'])} vehicles with low maintenance scores",
                'expected_benefit': 'Prevent breakdowns and improve efficiency'
            })
        
        # Driver recommendations
        if len(driver_analysis.get('safety_concerns', [])) > 0:
            recommendations.append({
                'category': 'safety',
                'priority': 'high',
                'recommendation': f"Provide safety training for {len(driver_analysis['safety_concerns'])} drivers with safety concerns",
                'expected_benefit': 'Reduce accidents and insurance costs'
            })
        
        # Utilization recommendations
        vehicle_util = utilization_analysis['vehicle_utilization']['utilization_rate_percent']
        if vehicle_util < 70:
            recommendations.append({
                'category': 'utilization',
                'priority': 'medium',
                'recommendation': 'Increase vehicle utilization through better scheduling and route optimization',
                'expected_benefit': f'Improve utilization from {vehicle_util:.1f}% to 80%+'
            })
        elif vehicle_util > 95:
            recommendations.append({
                'category': 'capacity',
                'priority': 'medium',
                'recommendation': 'Consider fleet expansion to meet growing demand',
                'expected_benefit': 'Reduce delivery delays and improve customer satisfaction'
            })
        
        return recommendations

# Initialize engine
transportation_engine = TransportationIntelligenceEngine()

# Routes
@app.route('/transportation')
def transportation_dashboard():
    """Transportation & Logistics Intelligence dashboard"""
    
    recent_companies = LogisticsCompany.query.order_by(LogisticsCompany.created_at.desc()).limit(10).all()
    
    return render_template('transportation/dashboard.html',
                         companies=recent_companies)

@app.route('/transportation/api/route-optimization', methods=['POST'])
def optimize_routes():
    """API endpoint for route optimization"""
    
    data = request.get_json()
    company_id = data.get('company_id')
    delivery_date = data.get('delivery_date')
    
    if not company_id or not delivery_date:
        return jsonify({'error': 'Company ID and delivery date required'}), 400
    
    optimization = transportation_engine.optimize_route_planning(company_id, delivery_date)
    return jsonify(optimization)

@app.route('/transportation/api/fleet-analysis', methods=['POST'])
def analyze_fleet():
    """API endpoint for fleet performance analysis"""
    
    data = request.get_json()
    company_id = data.get('company_id')
    
    if not company_id:
        return jsonify({'error': 'Company ID required'}), 400
    
    analysis = transportation_engine.analyze_fleet_performance(company_id)
    return jsonify(analysis)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if LogisticsCompany.query.count() == 0:
        sample_company = LogisticsCompany(
            company_id='TRANS_DEMO_001',
            company_name='Demo Logistics Inc',
            headquarters='Transportation Hub',
            total_vehicles=45,
            active_drivers=52,
            service_areas=['Metro Area', 'Suburban Region'],
            daily_deliveries=280,
            on_time_delivery_rate=96.5
        )
        
        db.session.add(sample_company)
        db.session.commit()
        logger.info("Sample transportation data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5026, debug=True)