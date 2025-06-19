"""
Trading System Startup Script

This script provides an easy way to start the complete trading system
with all components properly initialized.
"""

import os
import sys
import time
import subprocess
import signal
from datetime import datetime
from pathlib import Path


class TradingSystemLauncher:
    """Manages the startup and shutdown of all trading system components."""
    
    def __init__(self):
        self.processes = {}
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
    def check_prerequisites(self):
        """Check if all required files and dependencies are present."""
        print("🔍 Checking prerequisites...")
        
        issues = []
        
        # Check required files
        required_files = [
            "live_trading.py",
            "signalr_market_hub.py",
            "projectx_connector.py",
            "order_manager.py",
            "telegram_notifier.py",
            "trading_monitor_v2.py",
            "indicator_calculation.py"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                issues.append(f"Missing file: {file}")
        
        # Check for model file
        model_files = list(Path(".").glob("*.pkl"))
        if not model_files:
            issues.append("No model file (*.pkl) found")
        
        # Check config
        try:
            from config import FUTURES
            if not FUTURES.get("topstep", {}).get("username"):
                issues.append("Missing broker credentials in config.py")
        except ImportError:
            issues.append("Cannot import config.py")
        
        if issues:
            print("\n❌ Found issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✅ All prerequisites met")
        return True
    
    def start_component(self, name, command, wait_time=2):
        """Start a system component in a subprocess."""
        try:
            log_file = self.log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    shell=True
                )
            
            self.processes[name] = {
                'process': process,
                'log_file': log_file
            }
            
            print(f"✅ Started {name} (PID: {process.pid}, Log: {log_file.name})")
            time.sleep(wait_time)  # Give component time to initialize
            
            # Check if process is still running
            if process.poll() is not None:
                print(f"❌ {name} failed to start! Check log: {log_file}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False
    
    def start_trading_system(self):
        """Start the main trading system."""
        print("\n🚀 Starting Trading System...")
        return self.start_component(
            "trading_system",
            f"{sys.executable} live_trading.py",
            wait_time=5  # Give more time for initialization
        )
    
    def start_monitor(self):
        """Start the trading monitor dashboard."""
        print("\n📊 Starting Trading Monitor...")
        
        # The monitor needs to wait for the trading system to be ready
        print("  Waiting for trading system to initialize...")
        time.sleep(10)
        
        return self.start_component(
            "trading_monitor",
            f"{sys.executable} trading_monitor_v2.py"
        )
    
    def check_telegram(self):
        """Check if Telegram is configured and offer to test."""
        print("\n📱 Checking Telegram configuration...")
        
        if os.path.exists("telegram_config.json"):
            print("✅ Telegram config found")
            
            response = input("Test Telegram notifications? (y/n): ")
            if response.lower() == 'y':
                subprocess.run([sys.executable, "helper_scripts.py", "telegram"])
        else:
            print("⚠️  Telegram not configured (optional)")
            print("   To enable: Create telegram_config.json with bot_token and chat_id")
    
    def monitor_processes(self):
        """Monitor running processes and restart if needed."""
        print("\n🔄 Monitoring system... Press Ctrl+C to stop all components")
        
        try:
            while True:
                # Check each process
                for name, info in self.processes.items():
                    process = info['process']
                    if process.poll() is not None:
                        print(f"\n⚠️  {name} stopped (exit code: {process.returncode})")
                        print(f"   Check log: {info['log_file']}")
                        
                        # Ask if should restart
                        response = input(f"Restart {name}? (y/n): ")
                        if response.lower() == 'y':
                            if name == "trading_system":
                                self.start_trading_system()
                            elif name == "trading_monitor":
                                self.start_monitor()
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down system...")
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all components."""
        for name, info in self.processes.items():
            process = info['process']
            if process.poll() is None:  # Still running
                print(f"Stopping {name}...")
                
                # Try graceful shutdown first
                if sys.platform == "win32":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
                
                # Wait up to 10 seconds
                try:
                    process.wait(timeout=10)
                    print(f"✅ {name} stopped")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    process.kill()
                    print(f"⚠️  {name} force killed")
        
        print("\n✅ All components stopped")
    
    def show_logs(self):
        """Display recent log entries."""
        print("\n📋 Recent logs:")
        for name, info in self.processes.items():
            log_file = info['log_file']
            if log_file.exists():
                print(f"\n--- {name} (last 10 lines) ---")
                with open(log_file) as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(line.rstrip())
    
    def run(self):
        """Main startup sequence."""
        print("="*60)
        print("🤖 TRADING SYSTEM LAUNCHER")
        print("="*60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Please fix the issues above before starting")
            return
        
        # Check Telegram
        self.check_telegram()
        
        print("\n" + "="*60)
        print("Starting components...")
        print("="*60)
        
        # Start main trading system
        if not self.start_trading_system():
            print("\n❌ Failed to start trading system")
            self.shutdown()
            return
        
        # Ask about monitor
        # response = input("\nStart trading monitor dashboard? (y/n): ")
        # if response.lower() == 'y':
        #     if not self.start_monitor():
        #         print("⚠️  Monitor failed to start, but trading system is running")
        
        # Show status
        print("\n" + "="*60)
        print("🟢 SYSTEM RUNNING")
        print("="*60)
        print("\nActive components:")
        for name, info in self.processes.items():
            print(f"  - {name}: PID {info['process'].pid}")
        
        print("\nOptions:")
        print("  - Press 'L' + Enter to show recent logs")
        print("  - Press Ctrl+C to stop all components")
        print("  - Monitor will auto-restart if it crashes")
        
        # Monitor processes
        self.monitor_processes()


def quick_start():
    """Quick start with minimal prompts."""
    launcher = TradingSystemLauncher()
    
    print("🚀 QUICK START MODE")
    print("="*40)
    
    if not launcher.check_prerequisites():
        return
    
    # Start everything automatically
    print("\nStarting all components...")
    
    if launcher.start_trading_system():
        launcher.start_monitor()
        
        print("\n✅ All systems go!")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            launcher.shutdown()
    else:
        print("❌ Startup failed")


def safe_mode():
    """Start in safe mode with extra checks."""
    print("🛡️  SAFE MODE")
    print("="*40)
    print("This mode performs extra checks before starting")
    
    # Run health check first
    subprocess.run([sys.executable, "helper_scripts.py", "health"])
    
    response = input("\nContinue with startup? (y/n): ")
    if response.lower() == 'y':
        launcher = TradingSystemLauncher()
        launcher.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading System Launcher")
    parser.add_argument("--quick", action="store_true", help="Quick start mode")
    parser.add_argument("--safe", action="store_true", help="Safe mode with extra checks")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_start()
    elif args.safe:
        safe_mode()
    else:
        launcher = TradingSystemLauncher()
        launcher.run()