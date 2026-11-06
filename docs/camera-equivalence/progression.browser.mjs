export default async function checkTransformProgression(page) {
  const check = (condition,message) => { if (!condition) throw new Error(message); };
  const panel = page.getByRole("region",{name:"Transform progression"});
  const positions = selector => panel.locator(selector).evaluateAll(nodes =>
    nodes.map(p => [p.getAttribute("cx"),p.getAttribute("cy")]));
  await panel.waitFor();
  check(await panel.getByRole("slider").count() === 1,"One slider controls all three views");
  check(await panel.locator(".image-view").count() === 2,"Both camera views accompany the 3D construction");
  const source = await positions("[data-transform-source]");
  const original = await page.locator('[data-scene-mode="world"] svg').innerHTML();
  const world = panel.locator('[data-scene-mode="progression-world"]');
  let previousWorld = await world.locator("svg").innerHTML();
  let previousTargets = await positions("[data-transform-target]");
  for (let stage=1;stage<=4;stage++) {
    await panel.locator(`[data-transform-stage="${stage}"]`).click();
    check(await panel.getAttribute("data-transform-family") === String(stage-1),`Milestone ${stage} selects its exact family`);
    const outputs = await panel.locator(".model-errors output").allTextContents();
    outputs.forEach((value,i) => check((value === "exact") === (i >= stage-1),`Milestone ${stage}: model ${i} exactness`));
    check(JSON.stringify(await positions("[data-transform-source]")) === JSON.stringify(source),"Source rays stay fixed");
    const updatedWorld = await world.locator("svg").innerHTML();
    const updatedTargets = await positions("[data-transform-target]");
    check(updatedWorld !== previousWorld,"Each milestone updates the 3D geometry");
    check(JSON.stringify(updatedTargets) !== JSON.stringify(previousTargets),"Each milestone updates the image-2 projections");
    previousWorld = updatedWorld;
    previousTargets = updatedTargets;
  }
  check(await page.locator('[data-scene-mode="world"] svg').innerHTML() === original,"The experiment leaves the main world controls intact");
  check(JSON.stringify(await positions("[data-transform-target]")) !== JSON.stringify(await positions("[data-transform-predicted]")),
    "Off-plane samples separate from plane predictions");
  await panel.locator("#transform-complexity").focus();
  await page.keyboard.press("Home");
  check(await panel.getAttribute("data-transform-family") === "0","The slider supports keyboard reset");
  check(await world.locator('[data-point^="construction-"]').count() === 25,"The 3D construction contains every image sample");
  await world.getByRole("button",{name:"Reset view"}).click();
  const plots = () => panel.locator(".image-view svg").evaluateAll(nodes => nodes.map(node => node.innerHTML));
  const resetPlots = await plots();
  await page.locator("#reader-azimuth").evaluate(input => {
    input.value = "-35";
    input.dispatchEvent(new Event("input",{bubbles:true}));
  });
  check(JSON.stringify(await plots()) === JSON.stringify(resetPlots),"Reader motion does not alter either camera image");
  await world.getByRole("button",{name:"Reset view"}).click();
  const geometry = await world.locator("svg").innerHTML();
  await world.locator("svg").scrollIntoViewIfNeeded();
  const box = await world.locator("svg").boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);
  await page.mouse.down();
  await page.mouse.move(box.x+box.width/2+30,box.y+box.height/2+16,{steps:4});
  await page.mouse.up();
  check(await world.locator("svg").innerHTML() !== geometry,"Dragging the shared scene rotates the reader");
  check(JSON.stringify(await plots()) === JSON.stringify(resetPlots),"Scene dragging leaves both image projections fixed");
  await world.getByRole("button",{name:"Reset view"}).click();
  const viewport = await page.evaluate(() => ({width:innerWidth,height:innerHeight}));
  if (viewport.width >= 390 && viewport.height >= 800) {
    const top = viewport.width < 1280 ? 112 : 64;
    await panel.evaluate((element,top) => window.scrollTo(0,scrollY+element.getBoundingClientRect().top-top),top);
    const controls = await panel.locator(".progression-controls").boundingBox();
    check(controls.y >= top-1 && controls.y+controls.height <= viewport.height,"The complete shared controls stay below the page header");
    const scene = await world.locator(".scene-viewport").boundingBox();
    check(scene.y >= controls.y+controls.height && scene.y+scene.height <= viewport.height-64,
      "The 3D scene stays visible below the shared controls");
    for (const image of await panel.locator(".image-view svg").all()) {
      const bounds = await image.boundingBox();
      check(bounds.x >= 0 && bounds.x+bounds.width <= viewport.width &&
        bounds.y >= scene.y+scene.height && bounds.y+bounds.height <= viewport.height-64,
      "Both camera images stay visible above the document-review toolbar");
    }
  }
}
