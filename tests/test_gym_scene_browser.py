"""Native geometry, sensor debug and camera controls in the Three renderer."""

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page

pytestmark = pytest.mark.slow


@pytest.fixture
def scene_page(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#gym')
    page.wait_for_function("document.querySelector('gym-scene')?.renderer !== undefined")
    page.evaluate("""() => {
        const scene = document.querySelector('gym-scene');
        scene.hidden = false;
        scene.update({environment:'car', heading_unit:'radians',
          roads:[{id:'native-curve',width:3.5,points:[[0,0],[15,0],[30,4],[40,15],[42,35]]}],
          vehicle:{id:'ego'}, vehicles:[{id:'ego',position:[8,0],heading:0,
            length:4.6,width:1.9,height:1.6,last_action:[.25,.7]}],
          lidar:{origin:[8,0],endpoints:[[18,0],[8,30]],distances_m:[10,30],range_m:30},
          stop_signs:[{id:'route-stop',position:[30,7],heading:0,heading_unit:'radians',stop_line:[[28,2],[28,5]]}]});
    }""")
    return page, errors


def test_sensor_endpoints_hits_native_heading_and_physical_materials(scene_page):
    page, errors = scene_page
    result = page.evaluate("""() => {
        const s=document.querySelector('gym-scene'), vehicle=s.meshes.get('vehicle-ego');
        const rays=s.meshes.get('sensor-rays').geometry;
        return {rays:Number(s.dataset.rayCount), hits:Number(s.dataset.hitCount),
          endpoints:Array.from(rays.getAttribute('position').array),
          heading:vehicle.userData.heading, target:vehicle.userData.target.toArray(),
          paint:vehicle.children[0].material.type,clearcoat:vehicle.children[0].material.clearcoat,
          scale:vehicle.scale.toArray(),control:s.meshes.has('control-ego'),
          stop:s.meshes.has('stop-route-stop'),line:s.meshes.has('stop-line-route-stop')};
    }""")
    assert result['rays'] == 2 and result['hits'] == 1
    assert result['endpoints'] == pytest.approx([8,.65,0,18,.65,0,8,.65,0,8,.65,30])
    assert result['heading'] == pytest.approx(1.5707963267948966)
    assert result['target'] == [8,0,0]
    assert result['paint'] == 'MeshPhysicalMaterial' and result['clearcoat'] == 1
    assert result['scale'] == pytest.approx([1.9/1.8,1.6/1.51,4.6/4.5])
    assert result['control'] and result['stop'] and result['line']
    assert not errors


def test_follow_orbit_sensor_toggle_and_whole_map_controls(scene_page):
    page, errors = scene_page
    page.wait_for_function("""() => {
      const s=document.querySelector('gym-scene');
      return s.camera.position.distanceTo(s.meshes.get('vehicle-ego').position)<25;
    }""")
    scene = page.locator('gym-scene')
    scene.get_by_label('Sensor rays', exact=True).uncheck()
    assert page.evaluate("!document.querySelector('gym-scene').meshes.has('sensor-rays')")
    scene.get_by_label('Sensor rays', exact=True).check()
    assert page.evaluate("document.querySelector('gym-scene').dataset.rayCount==='2'")
    scene.get_by_label('Heading / control', exact=True).uncheck()
    assert page.evaluate("!document.querySelector('gym-scene').meshes.has('heading-ego')")
    scene.get_by_role('button',name='Whole map',exact=True).click()
    assert not scene.get_by_label('Follow car',exact=True).is_checked()
    scene.get_by_label('Follow car',exact=True).check()
    page.wait_for_function("document.querySelector('gym-scene').follow")
    assert not errors
