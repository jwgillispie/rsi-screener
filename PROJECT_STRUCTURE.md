# Project Structure Overview

## 📁 Clean Organization

The project has been reorganized into a professional, modular structure:

### 🏗️ **Before** (Messy)
```
rsi-screener/
├── api_server.py          # Mixed in root
├── iron_condor/           # Core logic
├── iron_condor_app/       # Flutter files mixed
├── backtest_results/      # Data scattered
├── config.ini            # Config in root
├── *.py                  # Scripts everywhere
└── docs mixed with code
```

### ✨ **After** (Clean)
```
rsi-screener/
├── backend/               # 🐍 Python Backend
│   ├── api/              # FastAPI server
│   ├── core/             # Trading logic
│   ├── config/           # Configuration
│   ├── data/             # Results & cache
│   ├── scripts/          # CLI tools
│   └── tests/            # Test files
├── frontend/             # 📱 Flutter App
│   ├── lib/              # Dart code
│   ├── android/ios/etc   # Platform files
│   └── pubspec.yaml      # Dependencies
├── scripts/              # 🚀 Startup scripts
│   ├── start_backend.sh
│   ├── start_frontend.sh
│   └── demo_setup.sh
└── docs/                 # 📚 Documentation
    ├── README.md
    ├── API_README.md
    └── FLUTTER_README.md
```

## 🎯 **Benefits of New Structure**

### 🔧 **Development**
- **Clear separation** of backend and frontend
- **Modular organization** for easy maintenance
- **Professional structure** following best practices
- **Easy navigation** and file location

### 🚀 **Deployment**
- **Independent deployment** of backend and frontend
- **Docker-ready** structure
- **Environment-specific** configurations
- **Scalable architecture**

### 📚 **Documentation**
- **Centralized docs** in `/docs` folder
- **Component-specific** documentation
- **Clean README** structure
- **API documentation** separation

### 🧪 **Testing**
- **Organized test** files in `/backend/tests`
- **Easy test discovery**
- **Separate unit/integration** tests
- **CI/CD ready** structure

## 🎮 **Quick Commands**

### Start Everything
```bash
# Backend API
./scripts/start_backend.sh

# Frontend App
./scripts/start_frontend.sh

# Full demo setup
./scripts/demo_setup.sh
```

### Development
```bash
# Backend development
cd backend/api && python api_server.py

# Frontend development  
cd frontend && flutter run

# Run tests
cd backend && python -m pytest tests/
```

## 📝 **File Updates Made**

### ✅ **Import Paths Updated**
- All Python imports use new structure
- Relative imports fixed
- Path configurations updated

### ✅ **Configuration Paths**
- Config files moved to `backend/config/`
- Data files moved to `backend/data/`
- Scripts updated to find configs

### ✅ **Startup Scripts**
- New organized startup scripts
- Platform-specific Flutter launching
- Automated environment setup

### ✅ **Documentation**
- Comprehensive README structure
- Component-specific guides
- API documentation organized

The project is now professionally organized and ready for development, deployment, and maintenance! 🎉