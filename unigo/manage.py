#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    # Load .env file from project root (two levels up from manage.py)
    from pathlib import Path
    import sys
    
    # Get project root directory (frontend/)
    current_dir = Path(__file__).resolve().parent  # unigo/
    project_root = current_dir.parent  # frontend/
    env_file = project_root / '.env'
    
    # Load environment variables from .env
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"✅ Loaded environment variables from: {env_file}")
            
            # Add PROJECT_ROOT to sys.path if specified in .env
            project_root_env = os.getenv('PROJECT_ROOT')
            if project_root_env and project_root_env not in sys.path:
                sys.path.insert(0, project_root_env)
                print(f"✅ Added to PYTHONPATH: {project_root_env}")
            elif not project_root_env:
                # Fallback: use calculated project_root
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                    print(f"✅ Added to PYTHONPATH (auto-detected): {project_root}")
        except ImportError:
            print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
            print(f"⚠️  Attempting to use auto-detected project root: {project_root}")
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
    else:
        print(f"⚠️  .env file not found at: {env_file}")
        print(f"⚠️  Using auto-detected project root: {project_root}")
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'unigo.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
