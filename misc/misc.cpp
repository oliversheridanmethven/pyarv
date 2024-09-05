#include <vector>

template<typename Self>
class Base {
protected:
    using my_type = int;
    std::vector<my_type> bar; // <- I will be doing something special to hash this...

public:
    virtual void baz(void) const = 0;

    Self foo(void) const {
        Self result = *dynamic_cast<const Self *>(this);
        // do something with the private bar variable.
        return result;
    };
};

class Derived_A final : public Base<Derived_A> {
public:
    void baz(void) const override {}
};

class Derived_B : public Base<Derived_B> {
public:
    void baz(void) const override {}
};

template<>
struct std::hash<Derived_B> {
    std::size_t operator()(const auto &shape) const noexcept { return 0; }
};

template<typename Self> requires std::is_base_of_v<Base<Self>, Self>
struct std::hash<Self> {
    std::size_t operator()(const Self &shape) const noexcept { return 0; }
};

int main() {
    Derived_A d_a;
    Derived_B d_b;
    auto d_a_ = d_a.foo();
    d_a.baz();
    std::hash<Derived_B>{}(d_b);
    std::hash<Derived_A>{}(d_a);
}


//#include <vector>
//
//template<typename Self>
//class Base {
//protected:
//    std::vector<int> bar;
//public:
//    virtual void baz(void) const = 0;
//
//    Self foo(void) const {
//        Self result = *dynamic_cast<const Self *>(this);
//        // do something with the private bar variable.
//        return result;
//    };
//};
//
//class Derived final : public Base<Derived> {
//public:
//    virtual void baz(void) const override {
//        // e.g. do something with bar.
//    }
//};
//
//int main() {
//    Derived d;
//    auto d2 = d.foo();
//}
