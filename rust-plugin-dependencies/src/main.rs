use std::any::Any;
use std::collections::HashMap;

trait Plugin {
    fn do_something(&self);
    fn as_any(&self) -> &dyn Any;
    fn name(&self) -> &str;
    fn dependencies(&self) -> Vec<String> {
        Vec::new()
    }
}

struct Plugin1 {}

struct Plugin2 {}

struct System {
    plugins: HashMap<String, Box<dyn Plugin>>,
}

impl Plugin for Plugin1 {
    fn do_something(&self) {
        println!("ConcretePlugin1");
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "Plugin1"
    }

    fn dependencies(&self) -> Vec<String> {
        vec!["Plugin2".to_string()]
    }
}

impl Plugin2 {
    fn do_something_special(&self) {
        println!("ConcretePlugin2 is very special.");
    }
}


impl Plugin for Plugin2 {
    fn do_something(&self) {
        println!("ConcretePlugin2");
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "Plugin2"
    }
}

fn main() {
    let plugin1 = Plugin1 {};
    let plugin2 = Plugin2 {};

    let mut system = System {
        plugins: HashMap::new(),
    };

    system.plugins.insert(plugin1.name().to_owned(), Box::new(plugin1));
    system.plugins.insert(plugin2.name().to_owned(), Box::new(plugin2));

    for (_, plugin) in system.plugins.iter() {
        // Call do something only if the plugin is of type Plugin2
        if plugin.as_any().is::<Plugin2>() {
            plugin.do_something();

            // Do something that only Plugin2 can do
            plugin.as_any().downcast_ref::<Plugin2>().unwrap().do_something_special();
        }
    }
}
