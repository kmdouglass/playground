use std::any::Any;

trait Plugin {
    fn do_something(&self);
    fn as_any(&self) -> &dyn Any;
}

struct Plugin1 {}

struct Plugin2 {}

struct System {
    plugins: Vec<Box<dyn Plugin>>,
}

impl Plugin for Plugin1 {
    fn do_something(&self) {
        println!("ConcretePlugin1");
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Plugin for Plugin2 {
    fn do_something(&self) {
        println!("ConcretePlugin2");
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn main() {
    let plugin1 = Plugin1 {};
    let plugin2 = Plugin2 {};

    let mut system = System {
        plugins: Vec::new(),
    };

    system.plugins.push(Box::new(plugin1));
    system.plugins.push(Box::new(plugin2));

    for plugin in system.plugins.iter() {
        // Call do something only if the plugin is of type Plugin2
        if plugin.as_any().is::<Plugin2>() {
            plugin.do_something();
        }
    }
}
