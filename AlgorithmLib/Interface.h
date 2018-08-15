#pragma once
#include <unordered_map>
#include <string>
#include <OAIdl.h>

//typedef std::unordered_map<std::string, VARIANT> ParameterTable;

struct Parameter : protected VARIANT
{
public:
    template <typename T>
    void AsPointer(const T* pPointer)
    {
        vt = VT_BYREF;
        byref = const_cast<void*>(reinterpret_cast<const void *>(pPointer));
    }

    template <typename T>
    void AsPointer(T* pPointer)
    {
        vt = VT_BYREF;
        byref = reinterpret_cast<void*>(pPointer);
    }

    template <typename T>
    const T* AsPointer() const
    {
        assert(vt == VT_BYREF);
        return reinterpret_cast<const T*>(const_cast<const void*>(byref));
    }

    template <typename T>
    T* AsPointer()
    {
        assert(vt == VT_BYREF);
        return reinterpret_cast<T*>(byref);
    }

    void AsInt(const int value)
    {
        vt = VT_INT;
        intVal = value;
    }

    int AsInt()
    {
        assert(vt == VT_INT);
        return intVal;
    }

    int AsInt() const
    {
        assert(vt == VT_INT);
        return intVal;
    }

    void AsFloat(const float value)
    {
        vt = VT_R4;
        fltVal = value;
    }

    float AsFloat()
    {
        assert(vt == VT_R4);
        return fltVal;
    }

    float AsFloat() const
    {
        assert(vt == VT_R4);
        return fltVal;
    }

    void AsDouble(const double value)
    {
        vt = VT_R8;
        dblVal = value;
    }

    double AsDouble()
    {
        assert(vt == VT_R8);
        return dblVal;
    }

    double AsDouble() const
    {
        assert(vt == VT_R8);
        return dblVal;
    }

    void AsBool(const bool value)
    {
        vt = VT_BOOL;
        boolVal = value;
    }

    bool AsBool()
    {
        assert(vt == VT_BOOL);
        return !!boolVal;
    }

    bool AsBool() const
    {
        assert(vt == VT_BOOL);
        return !!boolVal;
    }
};

using ParameterTable = std::unordered_map<std::string, Parameter>;

class IAlgorithm
{
public:
    enum Error : unsigned int
    {
        eSuccess = 0,
        eNotImplement = 1,
        eCount
    };
public:
    virtual ~IAlgorithm() {};

    virtual void    SetParameters(ParameterTable& params)   = 0;
    virtual Error   Process(void)                           = 0;
};


