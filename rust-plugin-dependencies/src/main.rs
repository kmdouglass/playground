use std::any::Any;
use std::collections::HashMap;

trait Plugin {
    fn do_something(&self);
    fn as_any(&self) -> &dyn Any;
    fn name(&self) -> &str;
    fn dependencies(&self) -> Vec<Box<dyn Plugin>> {
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

    fn dependencies(&self) -> Vec<Box<dyn Plugin>> {
        vec![Box::new(Plugin2 {})]
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
    let input_plugins: Vec<Box<dyn Plugin>> = vec![Box::new(plugin1)];

    // Check if Plugin1 has any dependencies
    // If it does, add it to the hashmap
    let mut plugins = HashMap::new();
    for plugin in input_plugins {
        // Get the dependencies of the plugin first; otherwise, we will lose the reference to the plugin
        let mut dependencies: Vec<Box<dyn Plugin>> = plugin.dependencies();

        // Insert the plugin into the hashmap
        plugins.entry(plugin.name().to_string()).or_insert(plugin);

        // Insert the dependencies into the hashmap if they don't already exist
        for dependency in dependencies.into_iter() {
            plugins
                .entry(dependency.name().to_string())
                .or_insert(dependency);
        }
    }

    let mut system = System { plugins: plugins };

    // Assert Plugin1 and Plugin2 are in the system
    assert!(system.plugins.contains_key("Plugin1"));
    assert!(system.plugins.contains_key("Plugin2"));

    for (_, plugin) in system.plugins.iter() {
        // Call do something only if the plugin is of type Plugin2
        if plugin.as_any().is::<Plugin2>() {
            plugin.do_something();

            // Do something that only Plugin2 can do
            plugin
                .as_any()
                .downcast_ref::<Plugin2>()
                .unwrap()
                .do_something_special();
        }
    }
}
